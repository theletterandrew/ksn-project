"""
stream_extraction_wbt.py
------------------------
Extracts a fully connected stream network for each drainage basin produced
by delineate_and_clip_basins.py. For each basin:

  1. Re-compute D8 flow direction (FDR) from the clipped basin DEM
  2. Re-compute flow accumulation (FAC) from the basin FDR — kept as output
  3. Threshold the FAC raster to a binary stream mask
  4. Skeletonize the mask to 1-pixel-wide centrelines (scikit-image)
  5. Trace each centreline pixel-chain into a LineString, ordered
     mouth->source using the FDR raster so downstream ends are always
     at index 0
  6. Collapse junction-pixel clusters from skeletonization artifacts
     so every branch point is a single node
  7. Write the stream network to basin_XXXX/streams_connected.gpkg
  8. Optionally extract the longest outlet-to-headwater path and write
     it to basin_XXXX/streams_longest_branch.gpkg
  9. Delete the intermediate per-basin FDR (FAC is kept for Ksn/CHI)

USAGE:
    1. Install dependencies:
       pip install rasterio numpy fiona shapely scikit-image networkx

    2. Run delineate_and_clip_basins.py first to produce per-basin DEMs.

    3. Run:
       python stream_extraction_wbt.py

Requirements:
    - rasterio
    - numpy
    - fiona
    - shapely
    - scikit-image  (for skeletonize)
    - networkx      (for segment merging)
    - WhiteboxTools executable at config.WBT_EXE
    - Completed delineate_and_clip_basins.py first (produces basin DEMs)
"""

import logging
import subprocess
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

WBT_EXE    = config.WBT_EXE
BASINS_DIR = config.DATA_BASINS  # Output of delineate_and_clip_basins.py

# Drainage area threshold (cells). At 2 m resolution:
#   500,000  cells = ~2 km²
#   1,000,000 cells = ~4 km²
#   2,500,000 cells = ~10 km²
THRESHOLD = config.STREAM_THRESHOLD

# Minimum number of skeleton pixels a segment must contain to be written.
# Removes single-pixel stubs and short noise branches.
MIN_PIXELS = config.MIN_PIXELS

# Minimum stream length in metres. Segments shorter than this are dropped
# after conversion to LineStrings (threshold is in real map units, not pixels).
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
        if seg_upstream[i]:
            seg_upstream[i] = sorted(set(seg_upstream[i]))
    # --- after the seg_upstream construction loop ---

    # Break bidirectional (mutual) links at junctions.
    # If i lists j as upstream AND j lists i as upstream, the shorter
    # segment is the tributary; remove the link that points the wrong way.
    for i in list(seg_upstream.keys()):
        for j in list(seg_upstream[i]):
            if i in seg_upstream.get(j, []):
                # Mutual link — keep the direction where the downstream
                # segment is longer (higher FAC at mouth is a proxy).
                fac_i = float(fac_arr[lines[i][0][0][0], lines[i][0][0][1]])
                fac_j = float(fac_arr[lines[j][0][0][0], lines[j][0][0][1]])
                if fac_i >= fac_j:
                    # i is more downstream → j should NOT list i as upstream
                    seg_upstream[j] = [k for k in seg_upstream[j] if k != i]
                else:
                    # j is more downstream → i should NOT list j as upstream
                    seg_upstream[i] = [k for k in seg_upstream[i] if k != j]
    # Note: we intentionally keep ALL upstream candidates for each segment.
    # best_upstream_path() explores all branches recursively and picks the
    # longest by geometry — pruning to highest-FAC here would prevent it
    # from finding the true longest path when a lower-FAC branch is longer.
    seg_lengths = [line.length for _, line in lines]
    memo = {}

    def best_upstream_path(seg_idx, active):
        # Returns (total_length_from_seg_to_best_headwater, [path indices]).
        if seg_idx in memo:
            return memo[seg_idx]
        if seg_idx in active:
            # Safety against rare topology cycles.
            return seg_lengths[seg_idx], [seg_idx]

        active.add(seg_idx)
        best_total = seg_lengths[seg_idx]
        best_path = [seg_idx]

        for up_idx in seg_upstream.get(seg_idx, []):
            up_total, up_path = best_upstream_path(up_idx, active)
            candidate_total = seg_lengths[seg_idx] + up_total
            if candidate_total > best_total:
                best_total = candidate_total
                best_path = [seg_idx] + up_path
            elif np.isclose(candidate_total, best_total):
                # Tie-breaker: prefer branch with higher upstream mouth FAC.
                cand_fac = float(fac_arr[lines[up_idx][0][0][0], lines[up_idx][0][0][1]])
                curr_next = best_path[1] if len(best_path) > 1 else None
                curr_fac = float("-inf")
                if curr_next is not None:
                    curr_fac = float(fac_arr[lines[curr_next][0][0][0], lines[curr_next][0][0][1]])
                if cand_fac > curr_fac:
                    best_path = [seg_idx] + up_path

        active.remove(seg_idx)
        memo[seg_idx] = (best_total, best_path)
        return memo[seg_idx]

    # Evaluate all possible downstream starts (supports multi-outlet networks)
    # and pick the globally longest connected branch.
    best_total = -1.0
    path_indices = []
    downstream_start_idx = None
    # Identify the true basin outlet as the single segment with the highest
    # FAC at its mouth.  In a properly delineated basin there is exactly one
    # outlet and it always has the greatest accumulated drainage area.  Using
    # FAC directly is more robust than topological heuristics (e.g. "mouth not
    # in any source set") which can fail when the main stem outlet is trimmed
    # by the border or length filters.
    true_outlet_idx = max(
        range(len(lines)),
        key=lambda i: float(fac_arr[lines[i][0][0][0], lines[i][0][0][1]])
    )
    logger.info(
        f"  True outlet: seg[{true_outlet_idx}] "
        f"mouth={lines[true_outlet_idx][0][0]} "
        f"FAC={float(fac_arr[lines[true_outlet_idx][0][0][0], lines[true_outlet_idx][0][0][1]]):.0f}"
    )

    # Start the longest-path search from the true outlet only.
    best_total, path_indices = best_upstream_path(true_outlet_idx, set())
    downstream_start_idx = true_outlet_idx

    if downstream_start_idx is None or not path_indices:
        logger.warning("  Could not determine a valid longest branch path.")
        return

    # Log the selected downstream start using the correct variable name.
    start_fac = float(fac_arr[lines[downstream_start_idx][0][0][0], lines[downstream_start_idx][0][0][1]])
    logger.info(
        f"  Selected downstream start: seg[{downstream_start_idx}] "
        f"mouth={lines[downstream_start_idx][0][0]} FAC={start_fac:.0f} "
        f"upstream_segs={seg_upstream.get(downstream_start_idx, [])}"
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
    for order, idx in enumerate(path_indices, start=1):
        seg, line = lines[idx]
        logger.info(f"  path seg[{idx}] order={order} mouth={seg[0]} source={seg[-1]} len={line.length:.1f}m")
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


def run_wbt(tool: str, args: dict, logger: logging.Logger) -> bool:
    """
    Runs a WhiteboxTools command. Returns True on success.
    args is a dict of parameter name -> value.
    """
    cmd = [str(WBT_EXE), f"--run={tool}"]
    for key, val in args.items():
        cmd.append(f"--{key}={val}")
    logger.info(f"Running: {tool}")
    logger.info(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.stdout:
            for line in result.stdout.strip().splitlines():
                logger.info(f"  WBT: {line}")
        if result.stderr:
            for line in result.stderr.strip().splitlines():
                logger.warning(f"  WBT ERR: {line}")
        if result.returncode != 0:
            logger.error(f"{tool} failed with return code {result.returncode}")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to run {tool}: {e}")
        return False


def extract_streams_for_basin(
    basin_dem: Path,
    basin_dir: Path,
    logger: logging.Logger
) -> bool:
    """
    Run the full stream extraction pipeline for a single basin.
    Returns True on success, False on failure.

    Inputs:
        basin_dem  — clipped basin DEM (e.g. basin_0001.tif)
        basin_dir  — output directory for this basin

    Outputs (written to basin_dir):
        fac.tif                   — flow accumulation (kept for Ksn/CHI)
        streams_connected.gpkg    — full stream network
        streams_longest_branch.gpkg — longest path (if EXTRACT_LONGEST_BRANCH)

    Intermediates (deleted after use):
        fdr.tif                   — per-basin flow direction
    """
    basin_dir.mkdir(parents=True, exist_ok=True)

    fdr_path = basin_dir / "fdr.tif"
    fac_path = basin_dir / "fac.tif"
    out_path = basin_dir / "streams_connected.gpkg"
    out_longest_path = basin_dir / "streams_longest_branch.gpkg"

    start_time = time.time()

    try:
        # ── Step 1: Compute per-basin FDR ─────────────────────────────────────
        logger.info("  Step 1: Computing per-basin flow direction...")
        success = run_wbt("D8Pointer", {
            "dem":    str(basin_dem),
            "output": str(fdr_path),
        }, logger)
        if not success:
            logger.error("  D8Pointer failed — skipping basin.")
            return False

        # ── Step 2: Compute per-basin FAC ─────────────────────────────────────
        logger.info("  Step 2: Computing per-basin flow accumulation...")
        success = run_wbt("D8FlowAccumulation", {
            "input":    str(fdr_path),
            "output":   str(fac_path),
            "out_type": "cells",
            "pntr":     "true",
        }, logger)
        if not success:
            logger.error("  D8FlowAccumulation failed — skipping basin.")
            return False

        # ── Step 3: Threshold FAC to stream mask ──────────────────────────────
        logger.info("  Step 3: Thresholding FAC to stream mask...")
        with rasterio.open(str(fac_path)) as src:
            fac_data  = src.read(1).astype(np.float32)
            transform = src.transform
            crs       = src.crs
            nodata    = src.nodata

        valid_mask  = (fac_data != nodata) if nodata is not None else np.ones_like(fac_data, dtype=bool)
        stream_mask = (fac_data >= THRESHOLD) & valid_mask

        if BORDER_CELLS > 0:
            b = BORDER_CELLS
            stream_mask[:b,  :] = False
            stream_mask[-b:, :] = False
            stream_mask[:,  :b] = False
            stream_mask[:, -b:] = False
            logger.info(f"  Blanked {b}-cell border to suppress edge artifacts")

        pixel_count = int(stream_mask.sum())
        logger.info(f"  Stream pixels above threshold: {pixel_count:,}")
        del fac_data, valid_mask

        if pixel_count == 0:
            logger.error("  No stream pixels at this threshold — skipping basin.")
            return False

        # ── Step 4: Read FDR ───────────────────────────────────────────────────
        logger.info("  Step 4: Reading FDR raster...")
        with rasterio.open(str(fdr_path)) as src:
            fdr_arr    = src.read(1)
            fdr_nodata = src.nodata

        # ── Step 5: Skeletonize ────────────────────────────────────────────────
        logger.info("  Step 5: Skeletonizing stream mask...")
        skeleton = skeletonize_stream_mask(stream_mask, logger)
        del stream_mask

        # ── Step 6: Build adjacency graph and trace segments ──────────────────
        logger.info("  Step 6: Tracing skeleton into line segments...")
        adj = build_adjacency(skeleton)

        junc_before = sum(1 for nbrs in adj.values() if len(nbrs) >= 3)
        adj, junction_remap = collapse_junction_clusters(adj)
        junc_after  = sum(1 for nbrs in adj.values() if len(nbrs) >= 3)
        if junc_before != junc_after:
            logger.info(
                f"  Junction clusters collapsed: {junc_before} -> {junc_after} junction pixels"
            )

        segments = trace_segments(adj)

        if junction_remap:
            segments = [
                [junction_remap.get(px, px) for px in chain]
                for chain in segments
            ]
        logger.info(f"  Raw segments traced: {len(segments):,}")

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
            logger.error("  No segments survived the minimum pixel filter — skipping basin.")
            return False

        # ── Step 7: Orient segments mouth->source ─────────────────────────────
        logger.info("  Step 7: Orienting segments mouth->source using FAC...")
        with rasterio.open(str(fac_path)) as _src:
            fac_arr_orient = _src.read(1).astype(np.float32)
        segments = [order_segment_by_fac(s, fac_arr_orient) for s in segments]
        del fac_arr_orient

        junction_pixels_set = {n for n, nbrs in adj.items() if len(nbrs) >= 3}
        before = len(segments)

        # Separate segments into those that drain off the raster edge and those
        # that don't.  For a basin clipped from a larger DEM the true outlet
        # SHOULD drain off the edge — we want to keep exactly that one segment
        # (the one with the highest FAC at its mouth = most upstream area) and
        # discard all other edge-draining segments as boundary artifacts.
        with rasterio.open(str(fac_path)) as _fac_src:
            fac_for_filter = _fac_src.read(1).astype(np.float32)

        edge_draining = []
        non_edge = []
        for s in segments:
            is_connector = s[0] in junction_pixels_set and s[-1] in junction_pixels_set
            if not is_connector and drains_off_edge(s, fdr_arr, fdr_nodata):
                edge_draining.append(s)
            else:
                non_edge.append(s)

        if edge_draining:
            # Keep the one true outlet — highest FAC at mouth pixel
            true_outlet = max(edge_draining, key=lambda s: float(fac_for_filter[s[0][0], s[0][1]]))
            discarded = len(edge_draining) - 1
            segments = non_edge + [true_outlet]
            if discarded:
                logger.info(f"  Kept 1 true outlet (highest FAC); removed {discarded} edge-draining artifact(s)")
        del fac_for_filter

        removed = before - len(segments)
        if removed:
            logger.info(f"  Removed {removed:,} edge-draining segments (boundary artifacts)")
        logger.info(f"  Segments after edge-drain filter: {len(segments):,}")

        if not segments:
            logger.error("  No segments survived the edge-drain filter — skipping basin.")
            return False

        # ── Step 8: Convert to LineStrings and write GeoPackage ───────────────
        logger.info("  Step 8: Writing LineStrings to GeoPackage...")

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

        lines = []
        for seg in segments:
            line = pixels_to_linestring(seg, transform)
            if line.length > 0:
                lines.append((seg, line))

        if MIN_STREAM_LENGTH_M > 0:
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
                    f"  Protected {protected:,} short connector segments from length filter"
                )
            lines = kept

        if not lines:
            logger.error("  No segments survived the minimum length filter — skipping basin.")
            return False

        count = 0
        with fiona.open(
            str(out_path), mode="w", driver="GPKG",
            schema=schema, crs=out_crs, layer="streams",
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

        # ── Step 9: Longest branch (optional) ─────────────────────────────────
        if EXTRACT_LONGEST_BRANCH:
            with rasterio.open(str(fac_path)) as _src:
                fac_arr_full = _src.read(1).astype(np.float32)
            extract_longest_branch(lines, fac_arr_full, adj, out_longest_path, out_crs, logger)
            del fac_arr_full

        # ── Cleanup: delete intermediate FDR ──────────────────────────────────
        if fdr_path.exists():
            fdr_path.unlink()
            logger.info(f"  Deleted intermediate FDR: {fdr_path.name}")

        elapsed = time.time() - start_time
        logger.info(f"  Done — {count:,} segments in {elapsed / 60:.1f} min")
        logger.info(f"  Streams : {out_path}")
        logger.info(f"  FAC     : {fac_path}")
        if EXTRACT_LONGEST_BRANCH:
            logger.info(f"  Longest : {out_longest_path}")

        return True

    except Exception as e:
        logger.error(f"  FAILED: {e}", exc_info=True)
        return False


def main():
    basins_dir = Path(BASINS_DIR)
    basins_dir.mkdir(parents=True, exist_ok=True)

    # Log to the basins directory
    log_path = basins_dir / "stream_extraction_wbt.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)

    # Find all per-basin DEMs
    # Discover basins by finding basin_XXXX subdirectories containing dem.tif
    basin_dirs = sorted([
        d for d in basins_dir.iterdir()
        if d.is_dir() and d.name.startswith("basin_") and (d / "dem.tif").exists()
    ])
    if not basin_dirs:
        logger.error(f"No basin_XXXX/dem.tif found in {basins_dir}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"Stream extraction — {len(basin_dirs)} basin(s) found")
    logger.info(f"Threshold        : {THRESHOLD:,} cells (~{THRESHOLD * 4 / 1e6:.1f} km² at 2m)")
    logger.info(f"Min pixels       : {MIN_PIXELS}")
    logger.info(f"Min stream length: {MIN_STREAM_LENGTH_M} m")
    logger.info(f"Border cells     : {BORDER_CELLS}")
    logger.info("=" * 60)

    total_start = time.time()
    succeeded, failed = [], []

    for basin_dir in basin_dirs:
        basin_name = basin_dir.name          # e.g. "basin_0001"
        basin_dem  = basin_dir / "dem.tif"

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Processing: {basin_name}")
        logger.info("=" * 60)

        ok = extract_streams_for_basin(basin_dem, basin_dir, logger)  # basin_dir is already the output dir
        if ok:
            succeeded.append(basin_name)
        else:
            failed.append(basin_name)

    elapsed_total = time.time() - total_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("ALL BASINS COMPLETE")
    logger.info(f"  Succeeded : {len(succeeded)}")
    logger.info(f"  Failed    : {len(failed)}")
    if failed:
        for name in failed:
            logger.warning(f"    FAILED: {name}")
    logger.info(f"  Total time: {elapsed_total / 60:.1f} minutes")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
