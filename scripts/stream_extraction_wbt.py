"""
stream_extraction_wbt.py
------------------------
Extracts a fully connected stream network from WhiteboxTools flow
accumulation and flow direction outputs. Produces a GeoPackage of
ordered LineString features (mouth->source) suitable for ksn analysis
and watershed delineation.

Pipeline:
  1. Threshold the FAC raster to a binary stream mask
  2. Skeletonize the mask to 1-pixel-wide centrelines (scikit-image)
  3. Trace each centreline pixel-chain into a LineString, ordered
     mouth->source using the FDR raster so downstream ends are always
     at index 0
  4. Collapse junction-pixel clusters from skeletonization artifacts
     so every branch point is a single node
  5. Write the full network to streams_connected.gpkg
  6. Optionally extract the longest outlet-to-headwater path and write
     it to streams_longest_branch.gpkg (set EXTRACT_LONGEST_BRANCH=True)

USAGE:
    1. Install dependencies:
       pip install rasterio numpy fiona shapely scikit-image networkx

    2. Edit the paths and threshold in the CONFIG section below.

    3. Run:
       python stream_extraction_wbt.py

Requirements:
    - rasterio
    - numpy
    - fiona
    - shapely
    - scikit-image  (for skeletonize)
    - networkx      (for segment merging)
    - Completed wbt_hydrology.py first (produces FAC + FDR rasters)
"""

import logging
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
import fiona
import fiona.crs
from shapely.geometry import LineString, mapping

# Calculate the path to the project root (one level up from scripts/)
root_dir = Path(__file__).resolve().parent.parent

# Add the root directory to sys.path so Python can find config.py
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

# =============================================================================
# CONFIG — Edit these before running
# =============================================================================

WBT_DIR     = config.DATA_SCRATCH_WBT
OUTPUT_DIR  = config.DATA_STREAMS

FAC_FILE    = "flow_accumulation.tif"
FDR_FILE    = "flow_direction.tif"
OUTPUT_FILE = "streams_connected.gpkg"

# Drainage area threshold (cells). At 2 m resolution:
#   500,000  cells = ~2 km²
#   1,000,000 cells = ~4 km²
#   2,500,000 cells = ~10 km²
THRESHOLD = config.STREAM_THRESHOLD

# Minimum number of skeleton pixels a segment must contain to be written.
# Removes single-pixel stubs and short noise branches.
MIN_PIXELS = 10

# Minimum stream length in metres. Segments shorter than this are dropped
# after conversion to LineStrings (threshold is in real map units, not pixels).
# Set in config.py as MIN_STREAM_LENGTH_M.
MIN_STREAM_LENGTH_M = config.MIN_STREAM_LENGTH_M

# Number of border cells to blank on all four edges before thresholding.
# Edge cells drain "off the raster" in D8 routing and accumulate spurious
# flow, creating false streams along the DEM boundary.
# At 2 m resolution, 3 cells = 6 m. Increase to 5-10 if artifacts persist.
BORDER_CELLS = config.BORDER_CELLS

# Extract the longest continuous mouth->headwater path from the network and
# save it as a separate GeoPackage.  Useful for ksn profile extraction along
# the main stem.  Uses a double-BFS on the segment graph so it runs in O(N).
EXTRACT_LONGEST_BRANCH = getattr(config, "EXTRACT_LONGEST_BRANCH", True)
LONGEST_BRANCH_FILE    = "streams_longest_branch.gpkg"

# =============================================================================
# END CONFIG — No edits needed below this line
# =============================================================================

# WBT D8 pointer: value -> (row_offset, col_offset)
D8_OFFSETS = {
    1:   ( 0,  1),   # E
    2:   ( 1,  1),   # SE
    4:   ( 1,  0),   # S
    8:   ( 1, -1),   # SW
    16:  ( 0, -1),   # W
    32:  (-1, -1),   # NW
    64:  (-1,  0),   # N
    128: (-1,  1),   # NE
}

# Reverse: for each (dr, dc) what FDR value points in that direction
D8_FROM_OFFSET = {v: k for k, v in D8_OFFSETS.items()}


def setup_logging(output_dir: Path) -> logging.Logger:
    log_path = output_dir / "stream_extraction_wbt.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def skeletonize_stream_mask(stream_mask: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """
    Thin the binary stream mask to 1-pixel-wide centrelines using
    scikit-image's morphological skeletonization.

    Returns a boolean array of the same shape.
    """
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        logger.error(
            "scikit-image is required for skeletonization: "
            "pip install scikit-image"
        )
        sys.exit(1)

    logger.info("  Skeletonizing stream mask (thinning to 1-pixel centrelines)...")
    skeleton = skeletonize(stream_mask)
    skel_pixels = int(skeleton.sum())
    logger.info(f"  Skeleton pixels: {skel_pixels:,}")
    return skeleton


def build_adjacency(skeleton: np.ndarray) -> dict:
    """
    Build a pixel-level adjacency graph from the skeleton.
    Returns {(r,c): [(r2,c2), ...]} for all skeleton pixels.
    """
    rows, cols = np.where(skeleton)
    skel_set   = set(zip(rows.tolist(), cols.tolist()))
    adj        = defaultdict(list)

    for r, c in skel_set:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in skel_set:
                    adj[(r, c)].append(nb)

    return dict(adj)


def collapse_junction_clusters(adj: dict) -> dict:
    """
    Morphological skeletonization often produces small clusters of 2-4
    mutually-adjacent pixels at branch points instead of a single junction
    pixel.  Every pixel in such a cluster has 3+ neighbours and they all
    connect to each other, creating spurious 2-3 pixel "segments" between
    junction pixels that fragment the network.

    This function collapses each such cluster into a single representative
    node (the centroid pixel, chosen by median row/col) and rewires all
    external neighbours to that node, returning a cleaned adjacency dict.

    Algorithm
    ---------
    1. Find all junction pixels (degree >= 3).
    2. Union-find: merge junctions that are direct neighbours of each other
       into clusters.
    3. For each cluster pick a representative (pixel closest to centroid).
    4. Rewrite the adjacency dict, replacing every cluster member with its
       representative and removing self-loops.
    """
    junctions = {n for n, nbrs in adj.items() if len(nbrs) >= 3}

    # --- Union-Find over junction pixels that are adjacent to each other ---
    parent = {j: j for j in junctions}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for j in junctions:
        for nb in adj.get(j, []):
            if nb in junctions:
                union(j, nb)

    # Group cluster members by root
    clusters = defaultdict(set)
    for j in junctions:
        clusters[find(j)].add(j)

    # Pick the representative: pixel nearest to the integer centroid
    remap = {}
    for root, members in clusters.items():
        if len(members) == 1:
            remap[next(iter(members))] = next(iter(members))
            continue
        cr = int(round(sum(r for r, c in members) / len(members)))
        cc = int(round(sum(c for r, c in members) / len(members)))
        rep = min(members, key=lambda p: (p[0] - cr) ** 2 + (p[1] - cc) ** 2)
        for m in members:
            remap[m] = rep

    if not any(v != k for k, v in remap.items()):
        # No clusters to collapse — return unchanged
        return adj, {}

    # Rewrite adjacency: replace every remapped node, drop self-loops and dups
    new_adj = defaultdict(set)
    cluster_members = set(remap.keys())

    for node, nbrs in adj.items():
        new_node = remap.get(node, node)
        for nb in nbrs:
            new_nb = remap.get(nb, nb)
            if new_node != new_nb:
                new_adj[new_node].add(new_nb)

    # Ensure all nodes present even if they have no neighbours after remap
    for node in adj:
        new_node = remap.get(node, node)
        if new_node not in new_adj:
            new_adj[new_node] = set()

    return {k: list(v) for k, v in new_adj.items()}, remap


def trace_segments(adj: dict) -> list:
    """
    Walk the adjacency graph and extract linear pixel chains (segments).

    A junction is any pixel with >=3 neighbours. Segments run between
    junctions (or dead-ends), tracing each branch exactly once.

    Returns a list of pixel chains: [[(r,c), (r,c), ...], ...]
    """
    # Classify nodes
    endpoints   = {n for n, nbrs in adj.items() if len(nbrs) == 1}
    junctions   = {n for n, nbrs in adj.items() if len(nbrs) >= 3}
    start_nodes = endpoints | junctions

    visited_edges = set()
    segments      = []

    def _trace(start, direction):
        chain = [start, direction]
        prev  = start
        curr  = direction

        while True:
            nbrs = [n for n in adj.get(curr, []) if n != prev]
            # Stop at a junction — include it so adjacent segments share
            # the same endpoint and the network is topologically connected.
            if curr in junctions:
                chain.append(curr)
                break
            # Stop at dead-end
            if not nbrs:
                break
            if len(nbrs) == 1:
                nxt = nbrs[0]
                edge = (min(curr, nxt), max(curr, nxt))
                if edge in visited_edges:
                    break
                visited_edges.add(edge)
                chain.append(nxt)
                prev, curr = curr, nxt
            else:
                # len(nbrs) > 1 but curr not classified as junction —
                # shouldn't happen post-collapse, but stop safely.
                break
        return chain

    # Trace from all endpoints and junctions
    for start in start_nodes:
        for nb in adj.get(start, []):
            edge = (min(start, nb), max(start, nb))
            if edge in visited_edges:
                continue
            visited_edges.add(edge)
            chain = _trace(start, nb)
            if len(chain) >= 2:
                segments.append(chain)

    # Catch any isolated loops not reachable from endpoints/junctions
    all_nodes = set(adj.keys())
    visited   = {n for seg in segments for n in seg}
    remaining = all_nodes - visited
    while remaining:
        start = next(iter(remaining))
        loop  = [start]
        curr  = adj[start][0] if adj[start] else None
        while curr and curr != start and curr in remaining:
            loop.append(curr)
            nxt = [n for n in adj.get(curr, []) if n != loop[-2]] if len(loop) > 1 else adj.get(curr, [])
            curr = nxt[0] if nxt else None
        segments.append(loop)
        remaining -= set(loop)

    return segments


def order_segment_by_fac(
    chain: list,
    fac_arr: np.ndarray,
) -> list:
    """
    Orient a pixel chain so it runs mouth->source (downstream->upstream)
    using flow accumulation values.

    The mouth (downstream) end always has higher FAC than the source
    (upstream) end — it has accumulated more drainage area.  This is
    unambiguous and works for trunk segments, tributaries, and junction-
    adjacent segments alike, avoiding the FDR exit-chain logic which fails
    when both ends of a segment happen to exit the chain (e.g. trunk
    segments whose junction pixels point downstream out of the chain at
    both ends).
    """
    fac_first = float(fac_arr[chain[0][0],  chain[0][1]])
    fac_last  = float(fac_arr[chain[-1][0], chain[-1][1]])

    if fac_last > fac_first:
        # Last pixel has more drainage area — it is downstream, so reverse
        return list(reversed(chain))
    return chain


def drains_off_edge(chain: list, fdr_arr: np.ndarray, fdr_nodata) -> bool:
    """
    Return True if the downstream end of the chain (chain[0]) has a D8
    pointer that exits the raster boundary.

    Used as a secondary filter to remove segments whose only reason to
    exist is draining off a raster edge rather than to a real outlet.
    """
    nrows, ncols = fdr_arr.shape
    r, c = chain[0]
    fdr_val = int(fdr_arr[r, c])
    if fdr_nodata is not None and fdr_val == int(fdr_nodata):
        return False
    offset = D8_OFFSETS.get(fdr_val)
    if offset is None:
        return False
    nr, nc = r + offset[0], c + offset[1]
    return not (0 <= nr < nrows and 0 <= nc < ncols)


def pixels_to_linestring(chain: list, transform) -> LineString:
    """Convert a pixel chain [(r,c),...] to a map-coordinate LineString."""
    coords = [
        (transform.c + (c + 0.5) * transform.a,
         transform.f + (r + 0.5) * transform.e)
        for r, c in chain
    ]
    return LineString(coords)



def extract_longest_branch(
    lines: list,
    fac_arr: np.ndarray,
    adj: dict,
    out_path: Path,
    out_crs,
    logger: logging.Logger,
) -> None:
    """
    Find the globally longest connected outlet->headwater path and write
    those segments to *out_path*.

    We first build a directed upstream graph using segment topology:
    segment j is upstream of segment i when j's mouth pixel touches i's
    source pixel (exactly or as an 8-neighbour).

    Then we compute the maximum cumulative length path in that directed
    graph (dynamic programming on DFS). This returns the true longest
    branch by geometry length, not merely the highest-FAC tributary.
    """
    if not lines:
        logger.warning("  No segments available for longest-branch extraction.")
        return

    logger.info("Step 7: Extracting longest mouth->headwater branch...")

    # Build lookup of segment mouths. Multiple segments can share a mouth
    # pixel at a junction, so this is one-to-many.
    mouth_to_seg_indices = defaultdict(list)
    for i, (seg, _) in enumerate(lines):
        mouth_to_seg_indices[seg[0]].append(i)

    # For each segment, find tributaries that connect at its SOURCE end.
    # Because segments are ordered mouth->source, walking upstream means
    # stepping from current source to segments whose mouth is that source.
    #
    # We accept both exact pixel matches and 8-neighbour matches to be robust
    # to tiny skeletonization artifacts around confluences.
    seg_upstream = defaultdict(list)
    for i, (seg_i, _) in enumerate(lines):
        source_i = seg_i[-1]
        candidate_mouths = {source_i} | set(adj.get(source_i, []))
        for mouth_px in candidate_mouths:
            for j in mouth_to_seg_indices.get(mouth_px, []):
                if j != i:
                    seg_upstream[i].append(j)

        # Deduplicate while preserving deterministic order.
        if seg_upstream[i]:
            seg_upstream[i] = sorted(set(seg_upstream[i]))

    # Find the global outlet: segment with highest-FAC mouth pixel
    outlet_seg_idx = max(
        range(len(lines)),
        key=lambda i: float(fac_arr[lines[i][0][0][0], lines[i][0][0][1]])
    )
    outlet_fac = float(fac_arr[lines[outlet_seg_idx][0][0][0], lines[outlet_seg_idx][0][0][1]])
    logger.info(
        f"  Selected downstream start: seg[{best_start}] "
        f"mouth={lines[best_start][0][0]} FAC={start_fac:.0f} "
        f"upstream_segs={seg_upstream.get(best_start, [])}"
    )
    logger.info(
        f"  Longest path: {best_total:.1f} m  ({best_total/1000:.2f} km)"
        f"  |  {len(path_indices)} segments  outlet->headwater"
    )

    schema = {
        "geometry": "LineString",
        "properties": {
            "seg_id":      "int",
            "order_along": "int",
            "length_m":    "float",
            "n_vertices":  "int",
        },
    }

    if out_path.exists():
        out_path.unlink()

    with fiona.open(
        str(out_path), mode="w", driver="GPKG",
        schema=schema, crs=out_crs, layer="longest_branch",
    ) as dst:
        for order, idx in enumerate(path_indices, start=1):
            seg, line = lines[idx]
            dst.write({
                "geometry": mapping(line),
                "properties": {
                    "seg_id":      idx + 1,
                    "order_along": order,
                    "length_m":    round(line.length, 2),
                    "n_vertices":  len(seg),
                },
            })

    logger.info(f"  Written to: {out_path}")


def main():
    wbt_dir    = Path(WBT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    fac_path         = wbt_dir    / FAC_FILE
    fdr_path         = wbt_dir    / FDR_FILE
    out_path         = output_dir / OUTPUT_FILE
    out_longest_path = output_dir / LONGEST_BRANCH_FILE

    for label, p in [("FAC", fac_path), ("FDR", fdr_path)]:
        if not p.exists():
            logger.error(f"{label} raster not found: {p}")
            logger.error("Run wbt_hydrology.py first.")
            sys.exit(1)

    logger.info(f"Threshold        : {THRESHOLD:,} cells (~{THRESHOLD * 4 / 1e6:.1f} km² at 2m)")
    logger.info(f"Min pixels       : {MIN_PIXELS}")
    logger.info(f"Min stream length: {MIN_STREAM_LENGTH_M} m")
    logger.info(f"Border cells     : {BORDER_CELLS}")
    logger.info("-" * 60)

    start_time = time.time()

    try:
        # ------------------------------------------------------------------
        # Step 1: Read FAC, apply threshold
        # ------------------------------------------------------------------
        logger.info("Step 1: Reading FAC raster and applying threshold...")
        with rasterio.open(str(fac_path)) as src:
            fac_data  = src.read(1).astype(np.float32)
            transform = src.transform
            crs       = src.crs
            nodata    = src.nodata

        valid_mask  = (fac_data != nodata) if nodata is not None else np.ones_like(fac_data, dtype=bool)
        stream_mask = (fac_data >= THRESHOLD) & valid_mask

        # FIX 1: Blank the raster border before thresholding.
        # Edge cells drain "off the raster" in D8 routing and accumulate
        # spurious flow, producing false streams along the DEM boundary.
        if BORDER_CELLS > 0:
            b = BORDER_CELLS
            stream_mask[:b,  :] = False   # top
            stream_mask[-b:, :] = False   # bottom
            stream_mask[:,  :b] = False   # left
            stream_mask[:, -b:] = False   # right
            logger.info(f"  Blanked {b}-cell border to suppress edge artifacts")

        pixel_count = int(stream_mask.sum())
        logger.info(f"  Stream pixels above threshold: {pixel_count:,}")
        del fac_data, valid_mask

        if pixel_count == 0:
            logger.error("No stream pixels at this threshold. Lower THRESHOLD in config.py.")
            sys.exit(1)

        # ------------------------------------------------------------------
        # Step 2: Read FDR raster
        # ------------------------------------------------------------------
        logger.info("Step 2: Reading FDR raster...")
        with rasterio.open(str(fdr_path)) as src:
            fdr_arr    = src.read(1)
            fdr_nodata = src.nodata

        # ------------------------------------------------------------------
        # Step 3: Skeletonize
        # ------------------------------------------------------------------
        logger.info("Step 3: Skeletonizing stream mask...")
        skeleton    = skeletonize_stream_mask(stream_mask, logger)
        del stream_mask

        # ------------------------------------------------------------------
        # Step 4: Build adjacency graph and trace segments
        # ------------------------------------------------------------------
        logger.info("Step 4: Tracing skeleton into line segments...")
        adj = build_adjacency(skeleton)

        # Collapse junction-pixel clusters produced by skeletonization.
        # Morphological thinning often leaves 2-4 mutually-adjacent pixels at
        # branch points instead of a single node, creating spurious 2-3 px
        # segments that fragment the network.  Collapsing them first ensures
        # every branch point is represented by exactly one pixel.
        junc_before = sum(1 for nbrs in adj.values() if len(nbrs) >= 3)
        adj, junction_remap = collapse_junction_clusters(adj)
        junc_after  = sum(1 for nbrs in adj.values() if len(nbrs) >= 3)
        if junc_before != junc_after:
            logger.info(
                f"  Junction clusters collapsed: {junc_before} -> {junc_after} junction pixels"
            )

        segments = trace_segments(adj)

        # Apply the junction remap to every pixel chain so that segment
        # endpoints use the canonical representative pixel, not the original
        # pre-collapse pixel.  Without this, seg[0]/seg[-1] won't match the
        # corresponding endpoint of the adjacent segment, fragmenting the
        # directed graph used for longest-branch extraction.
        if junction_remap:
            segments = [
                [junction_remap.get(px, px) for px in chain]
                for chain in segments
            ]
        logger.info(f"  Raw segments traced: {len(segments):,}")

        # Filter short stubs — but protect segments that connect two junction
        # pixels, since dropping them breaks network topology regardless of length.
        # A segment is a connector if BOTH endpoints are junction pixels
        # (degree >= 3 in the collapsed adjacency graph).
        junction_pixels = {n for n, nbrs in adj.items() if len(nbrs) >= 3}
        kept_segs, dropped_stubs, protected_connectors = [], 0, 0
        for s in segments:
            both_ends_are_junctions = s[0] in junction_pixels and s[-1] in junction_pixels
            if len(s) >= MIN_PIXELS or both_ends_are_junctions:
                kept_segs.append(s)
                if len(s) < MIN_PIXELS:
                    protected_connectors += 1
            else:
                dropped_stubs += 1
        segments = kept_segs
        logger.info(
            f"  Segments after min-pixel filter ({MIN_PIXELS}px): {len(segments):,} "
            f"(dropped {dropped_stubs} stubs, protected {protected_connectors} short connectors)"
        )

        if not segments:
            logger.error(
                "No segments survived the minimum pixel filter. "
                "Lower MIN_PIXELS or THRESHOLD."
            )
            sys.exit(1)

        # ------------------------------------------------------------------
        # Step 5: Orient each segment mouth->source using FAC
        # ------------------------------------------------------------------
        logger.info("Step 5: Orienting segments mouth->source using FAC...")
        # Re-read FAC for orientation (fac_data was freed after Step 1 to
        # save memory; we only need it briefly here and in Step 7)
        with rasterio.open(str(fac_path)) as _src:
            fac_arr_orient = _src.read(1).astype(np.float32)
        segments = [order_segment_by_fac(s, fac_arr_orient) for s in segments]
        del fac_arr_orient

        # FIX 2: Remove segments whose downstream end drains off the raster
        # edge — but ONLY if they are not junction connectors (both endpoints
        # are junction pixels).  Removing a connector that happens to exit the
        # edge would break network topology.
        junction_pixels_set = {n for n, nbrs in adj.items() if len(nbrs) >= 3}
        before = len(segments)
        kept_drain = []
        for s in segments:
            is_connector = s[0] in junction_pixels_set and s[-1] in junction_pixels_set
            if is_connector or not drains_off_edge(s, fdr_arr, fdr_nodata):
                kept_drain.append(s)
        segments = kept_drain
        removed = before - len(segments)
        if removed:
            logger.info(f"  Removed {removed:,} edge-draining segments (boundary artifacts)")
        logger.info(f"  Segments after edge-drain filter: {len(segments):,}")

        if not segments:
            logger.error(
                "No segments survived the edge-drain filter. "
                "Your study area may legitimately drain off all edges — "
                "set BORDER_CELLS = 0 and re-run if this is expected."
            )
            sys.exit(1)

        # ------------------------------------------------------------------
        # Step 6: Convert to LineStrings and write GeoPackage
        # ------------------------------------------------------------------
        logger.info("Step 6: Writing LineStrings to GeoPackage...")

        schema = {
            "geometry": "LineString",
            "properties": {
                "seg_id":     "int",
                "length_m":   "float",
                "n_vertices": "int",
            },
        }
        out_crs = crs.to_wkt() if crs else None

        if out_path.exists():
            out_path.unlink()

        # Convert all segments to LineStrings
        lines = []
        for seg in segments:
            line = pixels_to_linestring(seg, transform)
            if line.length > 0:
                lines.append((seg, line))

        # Apply length filter — but ONLY to headwater stubs (segments whose
        # upstream end has no further upstream neighbours).  Connector segments
        # that link two other segments must never be dropped regardless of
        # length, otherwise the network fragments into disconnected pieces and
        # the longest-branch graph loses its outlet nodes.
        if MIN_STREAM_LENGTH_M > 0:
            # Build a set of all pixels that appear as the MOUTH end of any
            # segment — i.e. pixels something drains into.  A segment whose
            # SOURCE end (seg[-1]) does NOT appear as a mouth end of another
            # segment is a true headwater stub; only those are length-filtered.
            mouth_pixels = {seg[0] for seg, _ in lines}

            kept, dropped_stub, protected = [], 0, 0
            for seg, line in lines:
                is_headwater_stub = seg[-1] not in mouth_pixels
                if is_headwater_stub and line.length < MIN_STREAM_LENGTH_M:
                    dropped_stub += 1
                else:
                    if not is_headwater_stub and line.length < MIN_STREAM_LENGTH_M:
                        protected += 1
                    kept.append((seg, line))

            if dropped_stub:
                logger.info(
                    f"  Removed {dropped_stub:,} headwater stubs shorter than "
                    f"{MIN_STREAM_LENGTH_M} m (kept {len(kept):,})"
                )
            if protected:
                logger.info(
                    f"  Protected {protected:,} short connector segments "
                    f"from length filter (topology preserved)"
                )
            lines = kept

        if not lines:
            logger.error(
                "No segments survived the minimum length filter. "
                "Lower MIN_STREAM_LENGTH_M in config.py."
            )
            sys.exit(1)

        count = 0
        with fiona.open(
            str(out_path),
            mode="w",
            driver="GPKG",
            schema=schema,
            crs=out_crs,
            layer="streams",
        ) as dst:
            batch = []
            for seg, line in lines:
                count += 1
                batch.append({
                    "geometry": mapping(line),
                    "properties": {
                        "seg_id":     count,
                        "length_m":   round(line.length, 2),
                        "n_vertices": len(seg),
                    },
                })
                if len(batch) >= 1000:
                    dst.writerecords(batch)
                    batch.clear()
            if batch:
                dst.writerecords(batch)

        # ------------------------------------------------------------------
        # Step 7: Longest branch extraction (optional)
        # ------------------------------------------------------------------
        if EXTRACT_LONGEST_BRANCH:
            # Re-read FAC for outlet detection (fac_data was freed after Step 1)
            with rasterio.open(str(fac_path)) as _src:
                fac_arr_full = _src.read(1).astype(np.float32)
            extract_longest_branch(lines, fac_arr_full, adj, out_longest_path, out_crs, logger)
            del fac_arr_full

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("COMPLETE")
        logger.info(f"  Network   : {out_path}")
        logger.info(f"  Segments  : {count:,}")
        if EXTRACT_LONGEST_BRANCH:
            logger.info(f"  Longest   : {out_longest_path}")
        logger.info(f"  Total time: {elapsed / 60:.1f} minutes")
        logger.info("")
        logger.info("Load the .gpkg files in QGIS or ArcGIS Pro to verify.")
        logger.info("Each feature is a LineString ordered mouth->source.")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
