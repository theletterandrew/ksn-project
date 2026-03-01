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
        return adj

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

    return {k: list(v) for k, v in new_adj.items()}


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


def order_segment_by_fdr(
    chain: list,
    fdr_arr: np.ndarray,
    fdr_nodata,
) -> list:
    """
    Orient a pixel chain so it runs mouth->source (downstream->upstream).

    The downstream end is the pixel whose D8 pointer exits the chain
    (points to a pixel NOT in the chain, or to nodata/raster boundary).
    If both ends qualify (or neither does), keep the original order.
    """
    chain_set = set(map(tuple, chain))
    nrows, ncols = fdr_arr.shape

    def _exits_chain(pixel):
        r, c = pixel
        fdr_val = int(fdr_arr[r, c])
        if fdr_nodata is not None and fdr_val == int(fdr_nodata):
            return False
        offset = D8_OFFSETS.get(fdr_val)
        if offset is None:
            return False
        nr, nc = r + offset[0], c + offset[1]
        # Exits if next cell is outside raster or not in the chain
        if not (0 <= nr < nrows and 0 <= nc < ncols):
            return True
        return (nr, nc) not in chain_set

    first_exits = _exits_chain(chain[0])
    last_exits  = _exits_chain(chain[-1])

    if last_exits and not first_exits:
        # Last pixel is downstream — reverse so mouth is at index 0
        return list(reversed(chain))
    # first_exits (or ambiguous) — keep as-is
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



def build_directed_segment_graph(lines: list):
    """
    Build a *directed* node graph from (seg, LineString) pairs using the
    integer pixel coordinates from the raw pixel chains, not the floating-
    point map coordinates.

    Pixel coordinates are exact integers so there is no float-jitter problem.
    Segments are already ordered mouth->source (seg[0] = downstream pixel),
    so each segment contributes one directed edge:
        mouth_pixel  -->  source_pixel

    Returns
    -------
    upstream   : {pixel: [(upstream_pixel, seg_index, length_m), ...]}
    downstream : {pixel: [(downstream_pixel, seg_index, length_m), ...]}
    mouth_nodes: set of pixels that are true outlets (no downstream inflow)
    """
    upstream   = defaultdict(list)
    downstream = defaultdict(list)
    all_nodes  = set()

    for i, (seg, line) in enumerate(lines):
        mouth  = seg[0]   # (row, col) — downstream end
        source = seg[-1]  # (row, col) — upstream / headwater end
        length = line.length
        upstream[mouth].append((source, i, length))
        downstream[source].append((mouth, i, length))
        all_nodes.update([mouth, source])

    # True outlets: pixels nothing flows *into* from downstream
    mouth_nodes = all_nodes - set(downstream.keys())

    return dict(upstream), dict(downstream), mouth_nodes


def _longest_upstream_path(outlet, upstream_graph):
    """
    Find the longest path from *outlet* to any headwater by traversing the
    directed upstream graph.  Uses iterative DP (topological-order relaxation)
    so it is exact for DAGs (which a D8 stream network always is).

    Returns
    -------
    best_headwater : the node at the far end of the longest path
    dist           : {node: longest distance from outlet to that node}
    prev           : {node: (parent_node, seg_index)} for path reconstruction
    """
    # Iterative BFS/DP: process nodes in upstream order (BFS layers from outlet)
    dist = {outlet: 0.0}
    prev = {outlet: None}
    queue = deque([outlet])
    best_headwater, max_dist = outlet, 0.0

    while queue:
        node = queue.popleft()
        for nb, seg_idx, length in upstream_graph.get(node, []):
            new_dist = dist[node] + length
            if nb not in dist or new_dist > dist[nb]:
                dist[nb] = new_dist
                prev[nb] = (node, seg_idx)
                queue.append(nb)
                if new_dist > max_dist:
                    max_dist = new_dist
                    best_headwater = nb

    return best_headwater, dist, prev


def extract_longest_branch(
    lines: list,
    out_path: Path,
    out_crs,
    logger: logging.Logger,
) -> None:
    """
    Find the longest flow-direction-coherent path (outlet -> headwater)
    in the stream network and write those segments to *out_path*.

    Unlike the previous double-BFS approach, this uses a *directed* graph
    so the path can never cross a junction — it only ever travels upstream
    along one tributary at each confluence, matching true stream topology.

    Algorithm
    ---------
    1. Build a directed graph: edges point mouth -> source (upstream).
    2. Identify outlet node(s): nodes with no downstream neighbours
       (nothing drains into them).
    3. For each outlet run a longest-path DP upstream (exact on DAGs).
    4. Keep the globally longest path across all outlets.
    5. Reconstruct and write the ordered segment list.
    """
    if not lines:
        logger.warning("  No segments available for longest-branch extraction.")
        return

    logger.info("Step 7: Extracting longest mouth->headwater branch...")

    upstream_graph, downstream_graph, mouth_nodes = build_directed_segment_graph(lines)

    if not upstream_graph:
        logger.warning("  Segment graph is empty — skipping longest branch.")
        return

    # If no clean outlet found (e.g. every node has a downstream neighbour
    # because all outlets were filtered), fall back to the node with the
    # lowest total upstream degree as a best-guess outlet.
    if not mouth_nodes:
        logger.warning(
            "  No clean outlet node found — all segment endpoints have "
            "downstream neighbours. Falling back to lowest-degree node."
        )
        mouth_nodes = {min(upstream_graph, key=lambda n: len(upstream_graph.get(n, [])))}

    logger.info(f"  Outlet nodes found: {len(mouth_nodes)}")

    # Run longest-path DP from every outlet, keep the global best
    best_headwater = None
    best_dist      = {}
    best_prev      = {}
    best_outlet    = None
    best_length    = 0.0

    for outlet in mouth_nodes:
        headwater, dist, prev = _longest_upstream_path(outlet, upstream_graph)
        path_length = dist.get(headwater, 0.0)
        logger.info(f"    outlet {outlet}: longest path = {path_length:.1f} m")
        if path_length > best_length:
            best_length    = path_length
            best_headwater = headwater
            best_dist      = dist
            best_prev      = prev
            best_outlet    = outlet

    logger.info(
        f"  Longest path: {best_length:.1f} m  ({best_length / 1000:.2f} km)"
        f"  |  outlet -> headwater"
    )

    # Reconstruct segment indices: walk prev[] from headwater back to outlet
    path_seg_indices = []
    cursor = best_headwater
    while best_prev.get(cursor) is not None:
        parent, seg_idx = best_prev[cursor]
        path_seg_indices.append(seg_idx)
        cursor = parent
    path_seg_indices.reverse()   # now ordered outlet -> headwater

    logger.info(f"  Path segments: {len(path_seg_indices)}")

    schema = {
        "geometry": "LineString",
        "properties": {
            "seg_id":      "int",
            "order_along": "int",    # 1 = most downstream
            "length_m":    "float",
            "n_vertices":  "int",
        },
    }

    if out_path.exists():
        out_path.unlink()

    with fiona.open(
        str(out_path),
        mode="w",
        driver="GPKG",
        schema=schema,
        crs=out_crs,
        layer="longest_branch",
    ) as dst:
        for order, idx in enumerate(path_seg_indices, start=1):
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
        adj = collapse_junction_clusters(adj)
        junc_after  = sum(1 for nbrs in adj.values() if len(nbrs) >= 3)
        if junc_before != junc_after:
            logger.info(
                f"  Junction clusters collapsed: {junc_before} -> {junc_after} junction pixels"
            )

        segments = trace_segments(adj)
        logger.info(f"  Raw segments traced: {len(segments):,}")

        # Filter short stubs
        segments = [s for s in segments if len(s) >= MIN_PIXELS]
        logger.info(f"  Segments after min-pixel filter ({MIN_PIXELS}px): {len(segments):,}")

        if not segments:
            logger.error(
                "No segments survived the minimum pixel filter. "
                "Lower MIN_PIXELS or THRESHOLD."
            )
            sys.exit(1)

        # ------------------------------------------------------------------
        # Step 5: Orient each segment mouth->source using FDR
        # ------------------------------------------------------------------
        logger.info("Step 5: Orienting segments mouth->source using FDR...")
        segments = [order_segment_by_fdr(s, fdr_arr, fdr_nodata) for s in segments]

        # FIX 2: Remove segments whose downstream end drains off the raster
        # edge. These are boundary artifacts — real watershed outlets that
        # legitimately exit the DEM will have been removed by BORDER_CELLS
        # already; anything surviving here is likely a false stream.
        before = len(segments)
        segments = [s for s in segments if not drains_off_edge(s, fdr_arr, fdr_nodata)]
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
            extract_longest_branch(lines, out_longest_path, out_crs, logger)

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
