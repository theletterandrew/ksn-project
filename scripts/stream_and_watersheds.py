"""
stream_and_watersheds.py
------------------------
Combined pipeline: given DEM, FAC, and FDR rasters, this script:

  1. Thresholds FAC to a binary stream mask (blanking border cells first)
  2. Skeletonizes the mask to 1-pixel-wide centrelines
  3. Collapses junction-pixel clusters from skeletonization artifacts
  4. Traces pixel chains into LineString segments ordered mouth->source
  5. Filters short stubs while protecting topology connectors
  6. Writes the full stream network to streams_connected.gpkg
  7. Optionally extracts the longest outlet-to-headwater path
  8. Identifies stream outlet / pour points (one per exiting trunk)
  9. Snaps pour points to the highest-FAC cell within a search radius
 10. Delineates watersheds via D8 BFS and writes watersheds.shp + pour points

INPUTS
------
  DEM_FILE  : filled DEM (used for validation only)
  FAC_FILE  : flow accumulation raster (WBT D8)
  FDR_FILE  : flow direction raster   (WBT D8 pointer values 1,2,4,8,16,32,64,128)

OUTPUTS
-------
  streams_connected.gpkg      - full stream network (LineStrings, mouth->source)
  streams_longest_branch.gpkg - longest continuous path (optional)
  pourpoints_final.shp        - snapped outlet pour points
  watersheds.shp              - delineated watershed polygons

USAGE
-----
  Edit the CONFIG section below, then:
      python stream_and_watersheds.py

Requirements: numpy, rasterio, geopandas, shapely, fiona, scikit-image, networkx
"""

import logging
import sys
import time
import warnings
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
import rasterio.transform
import geopandas as gpd
import fiona
import fiona.crs
from shapely.geometry import LineString, mapping, shape as shapely_shape

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

# =============================================================================
# CONFIG — edit these paths and parameters before running
# =============================================================================

# --- Input rasters ---
DEM_FILE = config.DATA_SCRATCH_WBT / "dem_filled.tif"
FAC_FILE = config.DATA_SCRATCH_WBT / "flow_accumulation.tif"
FDR_FILE = config.DATA_SCRATCH_WBT / "flow_direction.tif"

# --- Output directory ---
OUTPUT_DIR = config.DATA_STREAMS

# --- Stream extraction parameters ---
# FAC threshold (cells). How many upstream cells must drain into a point
# before it is considered a stream.  At 2 m resolution:
#   500,000 cells  ≈  2 km²
#   1,000,000 cells ≈  4 km²
STREAM_THRESHOLD = config.MIN_DRAINAGE_AREA_CELLS

# Minimum number of skeleton pixels a segment must have to be kept.
# Removes single-pixel stubs and very short noise branches.
MIN_PIXELS = config.MIN_PIXELS

# Minimum stream segment length in map units (metres).
# Headwater stubs shorter than this are dropped.  Set to 0 to disable.
MIN_STREAM_LENGTH_M = config.MIN_STREAM_LENGTH_M

# Number of border cells to blank before thresholding.  Edge cells accumulate
# spurious flow in D8 routing and produce false streams at the DEM boundary.
BORDER_CELLS = config.BORDER_CELLS

# Set True to also write the longest outlet->headwater path.
EXTRACT_LONGEST_BRANCH = True

# --- Watershed delineation parameters ---
# Same FAC threshold used to define the stream network for outlet detection.
# Usually should match STREAM_THRESHOLD.
MIN_WATERSHED_AREA_CELLS = STREAM_THRESHOLD

# Search radius (in cells) when snapping raw outlets to the highest-FAC
# stream cell nearby.  Prevents cross-tributary jumps.
SNAP_DISTANCE = config.SNAP_DISTANCE   # cells

# =============================================================================
# END CONFIG
# =============================================================================

# WBT D8 pointer value -> (row_offset, col_offset)
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
D8_FROM_OFFSET = {v: k for k, v in D8_OFFSETS.items()}


# =============================================================================
# Logging
# =============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    log_path = output_dir / "stream_and_watersheds.log"
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


# =============================================================================
# PART 1 — STREAM EXTRACTION
# =============================================================================

def skeletonize_stream_mask(stream_mask: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """Thin the binary stream mask to 1-pixel-wide centrelines."""
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        logger.error("scikit-image is required: pip install scikit-image")
        sys.exit(1)

    logger.info("  Skeletonizing stream mask...")
    skeleton = skeletonize(stream_mask)
    logger.info(f"  Skeleton pixels: {int(skeleton.sum()):,}")
    return skeleton


def build_adjacency(skeleton: np.ndarray) -> dict:
    """Build a pixel-level 8-connected adjacency graph from the skeleton."""
    rows, cols = np.where(skeleton)
    skel_set = set(zip(rows.tolist(), cols.tolist()))
    adj = defaultdict(list)
    for r, c in skel_set:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in skel_set:
                    adj[(r, c)].append(nb)
    return dict(adj)


def collapse_junction_clusters(adj: dict) -> tuple:
    """
    Collapse clusters of mutually-adjacent junction pixels (degree >= 3)
    into single representative nodes.  Returns (cleaned_adj, remap_dict).
    """
    junctions = {n for n, nbrs in adj.items() if len(nbrs) >= 3}

    # Union-Find
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

    # Group clusters
    clusters = defaultdict(set)
    for j in junctions:
        clusters[find(j)].add(j)

    # Pick representative: pixel closest to the cluster centroid
    remap = {}
    for members in clusters.values():
        if len(members) == 1:
            continue
        arr = np.array(list(members))
        centroid = arr.mean(axis=0)
        dists = np.linalg.norm(arr - centroid, axis=1)
        rep = members - {tuple(arr[int(np.argmin(dists))])}
        chosen = tuple(arr[int(np.argmin(dists))])
        for m in members:
            if m != chosen:
                remap[m] = chosen

    if not remap:
        return adj, remap

    # Rewrite adjacency
    new_adj = {}
    for node, nbrs in adj.items():
        new_node = remap.get(node, node)
        new_nbrs = list({remap.get(nb, nb) for nb in nbrs} - {new_node})
        existing = new_adj.get(new_node, [])
        combined = list({*existing, *new_nbrs})
        new_adj[new_node] = combined

    return new_adj, remap


def trace_segments(adj: dict) -> list:
    """
    Walk the adjacency graph and return a list of pixel chains (lists of
    (row, col) tuples).  Each chain runs between two endpoints (degree != 2).
    """
    endpoints = {n for n, nbrs in adj.items() if len(nbrs) != 2}
    if not endpoints:
        # Closed loop — pick any start
        endpoints = {next(iter(adj))}

    visited_edges = set()
    segments = []

    for start in endpoints:
        for nb in adj.get(start, []):
            edge_key = (min(start, nb), max(start, nb))
            if edge_key in visited_edges:
                continue
            # Walk the chain
            chain = [start, nb]
            visited_edges.add(edge_key)
            prev, cur = start, nb
            while cur not in endpoints:
                nexts = [n for n in adj.get(cur, []) if n != prev]
                if not nexts:
                    break
                nxt = nexts[0]
                ek = (min(cur, nxt), max(cur, nxt))
                if ek in visited_edges:
                    break
                visited_edges.add(ek)
                chain.append(nxt)
                prev, cur = cur, nxt
            segments.append(chain)

    return segments


def order_segment_by_fac(seg: list, fac_arr: np.ndarray) -> list:
    """Ensure seg[0] is the downstream (highest-FAC) end."""
    if not seg:
        return seg
    fac_head = float(fac_arr[seg[0][0], seg[0][1]])
    fac_tail = float(fac_arr[seg[-1][0], seg[-1][1]])
    return seg if fac_head >= fac_tail else seg[::-1]


def pixels_to_linestring(seg: list, transform) -> LineString:
    """Convert a pixel chain to a projected LineString."""
    rows = [p[0] for p in seg]
    cols = [p[1] for p in seg]
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    return LineString(zip(xs, ys))


def drains_off_edge(seg: list, fdr_arr: np.ndarray, fdr_nodata) -> bool:
    """Return True if the mouth end (seg[0]) of a segment drains off the raster edge."""
    if not seg:
        return False
    r, c = seg[0]
    nrows, ncols = fdr_arr.shape
    fdr_val = int(fdr_arr[r, c])
    if fdr_nodata is not None and fdr_val == int(fdr_nodata):
        return True
    if fdr_val == 0:
        return True
    offset = D8_OFFSETS.get(fdr_val)
    if offset is None:
        return True
    nr, nc = r + offset[0], c + offset[1]
    return not (0 <= nr < nrows and 0 <= nc < ncols)


def extract_longest_branch(lines, fac_arr, adj, out_path, out_crs, logger):
    """
    Find and write the longest continuous mouth->headwater path in the network
    using double-BFS on the segment graph (O(N)).
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not installed — skipping longest branch extraction.")
        return

    logger.info("  Building directed segment graph for longest-branch extraction...")

    G = nx.DiGraph()
    for i, (seg, line) in enumerate(lines):
        G.add_node(i, mouth=seg[0], source=seg[-1], length=line.length)

    # Connect segments: if source of A == mouth of B, add edge A->B
    mouth_index = defaultdict(list)
    for i, (seg, _) in enumerate(lines):
        mouth_index[seg[0]].append(i)

    for i, (seg, _) in enumerate(lines):
        for j in mouth_index.get(seg[-1], []):
            if i != j:
                G.add_edge(i, j)

    # Find the outlet node(s): no incoming edges (nobody drains into them)
    outlets = [n for n in G.nodes if G.in_degree(n) == 0]
    if not outlets:
        logger.warning("  No outlet nodes found in segment graph; skipping longest branch.")
        return

    # BFS from each outlet to find the farthest headwater
    best_length = 0
    best_path = []
    for outlet in outlets:
        q = deque([(outlet, [outlet], 0.0)])
        while q:
            node, path, length = q.popleft()
            seg_len = G.nodes[node]["length"]
            total = length + seg_len
            succs = list(G.successors(node))
            if not succs:
                if total > best_length:
                    best_length = total
                    best_path = path
            else:
                for s in succs:
                    q.append((s, path + [s], total))

    if not best_path:
        logger.warning("  Could not find a longest branch path.")
        return

    schema = {
        "geometry": "LineString",
        "properties": {"seg_id": "int", "length_m": "float"},
    }
    if out_path.exists():
        out_path.unlink()

    with fiona.open(str(out_path), mode="w", driver="GPKG",
                    schema=schema, crs=out_crs, layer="longest_branch") as dst:
        for idx, node_idx in enumerate(best_path):
            seg, line = lines[node_idx]
            dst.write({
                "geometry": mapping(line),
                "properties": {"seg_id": idx + 1, "length_m": round(line.length, 2)},
            })

    logger.info(
        f"  Longest branch: {len(best_path)} segment(s), "
        f"{best_length:.0f} m -> {out_path.name}"
    )


def run_stream_extraction(
    fac_path, fdr_path, output_dir, logger
) -> tuple:
    """
    Run the full stream extraction pipeline.
    Returns (lines, transform, crs) where lines = [(seg, LineString), ...].
    """
    out_path = output_dir / "streams_connected.gpkg"
    out_longest_path = output_dir / "streams_longest_branch.gpkg"

    # ------------------------------------------------------------------
    # Step 1: Load rasters and build stream mask
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading FAC raster and building stream mask...")
    with rasterio.open(str(fac_path)) as src:
        fac_data   = src.read(1).astype(np.float64)
        transform  = src.transform
        crs        = src.crs
        fac_nodata = src.nodata
        nrows, ncols = src.shape

    if fac_nodata is not None:
        fac_data[fac_data == fac_nodata] = 0.0

    if BORDER_CELLS > 0:
        fac_data[:BORDER_CELLS,  :]  = 0.0
        fac_data[-BORDER_CELLS:, :]  = 0.0
        fac_data[:,  :BORDER_CELLS]  = 0.0
        fac_data[:, -BORDER_CELLS:]  = 0.0
        logger.info(f"  Blanked {BORDER_CELLS} border cell(s) on each edge")

    stream_mask = fac_data >= STREAM_THRESHOLD
    logger.info(
        f"  Stream cells (FAC >= {STREAM_THRESHOLD:,}): "
        f"{int(stream_mask.sum()):,} of {nrows * ncols:,}"
    )
    if not stream_mask.any():
        logger.error(
            "No cells meet the stream threshold — lower STREAM_THRESHOLD."
        )
        sys.exit(1)

    del fac_data

    # ------------------------------------------------------------------
    # Step 2: Load FDR
    # ------------------------------------------------------------------
    logger.info("Step 2: Loading FDR raster...")
    with rasterio.open(str(fdr_path)) as src:
        fdr_arr    = src.read(1)
        fdr_nodata = src.nodata
    logger.info(f"  FDR shape: {fdr_arr.shape[1]}x{fdr_arr.shape[0]}")

    # ------------------------------------------------------------------
    # Step 3: Skeletonize
    # ------------------------------------------------------------------
    logger.info("Step 3: Skeletonizing...")
    skeleton = skeletonize_stream_mask(stream_mask, logger)
    del stream_mask

    # ------------------------------------------------------------------
    # Step 4: Build adjacency, collapse junction clusters, trace segments
    # ------------------------------------------------------------------
    logger.info("Step 4: Building adjacency graph and tracing segments...")
    adj = build_adjacency(skeleton)
    logger.info(f"  Skeleton nodes: {len(adj):,}")
    del skeleton

    adj, junction_remap = collapse_junction_clusters(adj)
    if junction_remap:
        logger.info(f"  Collapsed {len(junction_remap):,} junction pixels to representatives")

    segments = trace_segments(adj)
    if junction_remap:
        segments = [
            [junction_remap.get(px, px) for px in chain]
            for chain in segments
        ]
    logger.info(f"  Raw segments: {len(segments):,}")

    # Filter short stubs, protect topology connectors
    junction_pixels = {n for n, nbrs in adj.items() if len(nbrs) >= 3}
    kept, dropped_stubs, protected = [], 0, 0
    for s in segments:
        is_connector = s[0] in junction_pixels and s[-1] in junction_pixels
        if len(s) >= MIN_PIXELS or is_connector:
            kept.append(s)
            if len(s) < MIN_PIXELS:
                protected += 1
        else:
            dropped_stubs += 1
    segments = kept
    logger.info(
        f"  After min-pixel filter ({MIN_PIXELS}px): {len(segments):,} "
        f"(dropped {dropped_stubs} stubs, protected {protected} connectors)"
    )
    if not segments:
        logger.error("No segments after min-pixel filter. Lower MIN_PIXELS or STREAM_THRESHOLD.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 5: Orient mouth->source using FAC
    # ------------------------------------------------------------------
    logger.info("Step 5: Orienting segments mouth->source...")
    with rasterio.open(str(fac_path)) as src:
        fac_arr_orient = src.read(1).astype(np.float32)
    segments = [order_segment_by_fac(s, fac_arr_orient) for s in segments]
    del fac_arr_orient

    # Remove edge-draining segments (boundary artifacts), protect connectors
    before = len(segments)
    kept_drain = []
    for s in segments:
        is_connector = s[0] in junction_pixels and s[-1] in junction_pixels
        if is_connector or not drains_off_edge(s, fdr_arr, fdr_nodata):
            kept_drain.append(s)
    removed = before - len(kept_drain)
    if removed:
        logger.info(f"  Removed {removed:,} edge-draining segments")
    segments = kept_drain
    if not segments:
        logger.error(
            "No segments after edge-drain filter. "
            "If your study area genuinely drains off all edges, set BORDER_CELLS = 0."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 6: Convert to LineStrings and apply length filter
    # ------------------------------------------------------------------
    logger.info("Step 6: Converting to LineStrings...")
    lines = []
    for seg in segments:
        line = pixels_to_linestring(seg, transform)
        if line.length > 0:
            lines.append((seg, line))

    if MIN_STREAM_LENGTH_M > 0:
        mouth_pixels = {seg[0] for seg, _ in lines}
        kept_l, dropped_l, protected_l = [], 0, 0
        for seg, line in lines:
            is_headwater = seg[-1] not in mouth_pixels
            if is_headwater and line.length < MIN_STREAM_LENGTH_M:
                dropped_l += 1
            else:
                if not is_headwater and line.length < MIN_STREAM_LENGTH_M:
                    protected_l += 1
                kept_l.append((seg, line))
        if dropped_l:
            logger.info(
                f"  Removed {dropped_l:,} headwater stubs < {MIN_STREAM_LENGTH_M} m "
                f"(protected {protected_l} short connectors)"
            )
        lines = kept_l

    if not lines:
        logger.error("No segments after length filter. Lower MIN_STREAM_LENGTH_M.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 7: Write stream network GeoPackage
    # ------------------------------------------------------------------
    logger.info(f"Step 7: Writing {len(lines):,} segments to {out_path.name}...")
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

    with fiona.open(str(out_path), mode="w", driver="GPKG",
                    schema=schema, crs=out_crs, layer="streams") as dst:
        batch = []
        for idx, (seg, line) in enumerate(lines, start=1):
            batch.append({
                "geometry": mapping(line),
                "properties": {
                    "seg_id":     idx,
                    "length_m":   round(line.length, 2),
                    "n_vertices": len(seg),
                },
            })
            if len(batch) >= 1000:
                dst.writerecords(batch)
                batch.clear()
        if batch:
            dst.writerecords(batch)

    logger.info(f"  Written: {out_path}")

    # ------------------------------------------------------------------
    # Step 8 (optional): Longest branch
    # ------------------------------------------------------------------
    if EXTRACT_LONGEST_BRANCH:
        logger.info("Step 8: Extracting longest branch...")
        with rasterio.open(str(fac_path)) as src:
            fac_full = src.read(1).astype(np.float32)
        extract_longest_branch(lines, fac_full, adj, out_longest_path, out_crs, logger)
        del fac_full

    return lines, transform, crs


# =============================================================================
# PART 2 — POUR POINT & WATERSHED DELINEATION
# =============================================================================

def _next_cell(r, c, fdr_arr, fdr_nd):
    """Return the D8 downstream neighbour of (r, c), or None if terminal."""
    nrows, ncols = fdr_arr.shape
    fdr_val = int(fdr_arr[r, c])
    if fdr_val == 0:
        return None
    if fdr_nd is not None and fdr_val == int(fdr_nd):
        return None
    offset = D8_OFFSETS.get(fdr_val)
    if offset is None:
        return None
    nr, nc = r + offset[0], c + offset[1]
    if not (0 <= nr < nrows and 0 <= nc < ncols):
        return None
    if fdr_nd is not None and int(fdr_arr[nr, nc]) == int(fdr_nd):
        return None
    return (nr, nc)


def find_outlets(fac_arr, fdr_arr, fac_nd, fdr_nd, min_accum_cells, logger):
    """
    Find one outlet cell per stream trunk that exits the study area.
    Returns a list of (row, col) tuples.
    """
    nrows, ncols = fac_arr.shape
    stream_mask = fac_arr >= min_accum_cells
    stream_rows, stream_cols = np.where(stream_mask)

    if len(stream_rows) == 0:
        logger.error(f"No FAC cells >= {min_accum_cells:,}. Check MIN_WATERSHED_AREA_CELLS.")
        return []
    logger.info(f"  {len(stream_rows):,} stream cells at FAC >= {min_accum_cells:,}")

    # Step A: raw outlets — stream cells whose D8 neighbour is off-network
    raw_outlets = set()
    for r, c in zip(stream_rows.tolist(), stream_cols.tolist()):
        nxt = _next_cell(r, c, fdr_arr, fdr_nd)
        if nxt is None or not stream_mask[nxt[0], nxt[1]]:
            raw_outlets.add((r, c))
    logger.info(f"  {len(raw_outlets)} raw outlet(s) before deduplication")

    if not raw_outlets:
        logger.warning(
            "  No raw outlets — basin may be internally draining.\n"
            "  Falling back to highest-FAC stream cell."
        )
        best = int(np.argmax(fac_arr * stream_mask))
        return [(int(best // ncols), int(best % ncols))]

    # Step B: remove upstream duplicates — keep only the most-downstream outlet
    # per trunk by walking downstream from each raw outlet.
    redundant = set()
    for (r0, c0) in raw_outlets:
        r, c = r0, c0
        for _ in range(nrows * ncols):
            nxt = _next_cell(r, c, fdr_arr, fdr_nd)
            if nxt is None:
                break
            r, c = nxt
            if (r, c) in raw_outlets:
                # (r0, c0) drains into another raw outlet — it is redundant
                redundant.add((r0, c0))
                break

    final_outlets = [o for o in raw_outlets if o not in redundant]
    logger.info(
        f"  {len(final_outlets)} outlet(s) after deduplication "
        f"(removed {len(redundant)} upstream duplicates)"
    )
    return final_outlets


def snap_outlets_to_stream(outlets, fac_arr, fac_nd, transform,
                           snap_dist, min_accum_cells, logger):
    """
    Snap each outlet to the highest-FAC stream cell within snap_dist cells,
    capped so FAC cannot exceed 2× the outlet's current FAC (prevents
    cross-tributary jumps).
    """
    snapped = []
    for r0, c0 in outlets:
        fac0 = float(fac_arr[r0, c0])
        fac_cap = fac0 * 2.0

        r_lo = max(0, r0 - snap_dist)
        r_hi = min(fac_arr.shape[0], r0 + snap_dist + 1)
        c_lo = max(0, c0 - snap_dist)
        c_hi = min(fac_arr.shape[1], c0 + snap_dist + 1)

        window = fac_arr[r_lo:r_hi, c_lo:c_hi].copy()
        if fac_nd is not None:
            window[window == fac_nd] = 0.0

        # Mask: must be a stream cell and within the FAC cap
        stream_window = (window >= min_accum_cells) & (window <= fac_cap)
        if not stream_window.any():
            snapped.append((r0, c0))
            continue

        window[~stream_window] = 0.0
        local_idx = int(np.argmax(window))
        lr, lc = divmod(local_idx, window.shape[1])
        nr, nc = r_lo + lr, c_lo + lc

        if (nr, nc) != (r0, c0):
            logger.info(
                f"  Snapped ({r0},{c0}) FAC={fac0:,.0f} -> "
                f"({nr},{nc}) FAC={fac_arr[nr,nc]:,.0f}"
            )
        snapped.append((nr, nc))

    return snapped


def _d8_watershed_bfs(fdr_arr, fdr_nd, outlet_rows, outlet_cols):
    """
    Pure-numpy D8 BFS watershed delineation.
    Returns a label array (int32) where each cell is assigned the ID of its
    downstream outlet (1-based), or 0 for unassigned.
    """
    nrows, ncols = fdr_arr.shape
    D8_REVERSE = {
        ( 0,  1): 1,   ( 1,  1): 2,   ( 1,  0): 4,   ( 1, -1): 8,
        ( 0, -1): 16,  (-1, -1): 32,  (-1,  0): 64,  (-1,  1): 128,
    }
    # For each direction (dr, dc), what FDR value in the neighbour points TOWARD us
    UPSTREAM = [
        (dr, dc, D8_REVERSE[(-dr, -dc)])
        for (dr, dc) in D8_REVERSE
        if (-dr, -dc) in D8_REVERSE
    ]

    labels = np.zeros(fdr_arr.shape, dtype=np.int32)
    for ws_id, (out_r, out_c) in enumerate(zip(outlet_rows, outlet_cols), start=1):
        queue = deque([(out_r, out_c)])
        labels[out_r, out_c] = ws_id
        while queue:
            r, c = queue.popleft()
            for dr, dc, expected_fdr in UPSTREAM:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < nrows and 0 <= nc < ncols):
                    continue
                if labels[nr, nc] != 0:
                    continue
                cell_fdr = int(fdr_arr[nr, nc])
                if fdr_nd is not None and cell_fdr == int(fdr_nd):
                    continue
                if cell_fdr == expected_fdr:
                    labels[nr, nc] = ws_id
                    queue.append((nr, nc))
    return labels


def run_watershed_delineation(fac_path, fdr_path, dem_path, output_dir, logger):
    """
    Find pour points, snap them, delineate watersheds, write outputs.
    """
    # Validate inputs
    missing = [
        (lbl, p) for lbl, p in [
            ("DEM", dem_path), ("FAC", fac_path), ("FDR", fdr_path)
        ] if not p.exists()
    ]
    if missing:
        for lbl, p in missing:
            logger.error(f"  Missing input [{lbl}]: {p}")
        sys.exit(1)

    # Load FAC
    with rasterio.open(str(fac_path)) as src:
        cell_w    = abs(src.res[0])
        transform = src.transform
        fac_crs   = src.crs
        fac_arr   = src.read(1).astype(np.float64)
        fac_nd    = src.nodata
        if fac_nd is not None:
            fac_arr[fac_arr == fac_nd] = 0.0

    # Load FDR
    with rasterio.open(str(fdr_path)) as src:
        fdr_arr = src.read(1)
        fdr_nd  = src.nodata

    cell_area_km2 = (cell_w ** 2) / 1e6
    logger.info(
        f"FAC raster : {fac_arr.shape[1]}x{fac_arr.shape[0]} px @ {cell_w:.1f} m, "
        f"max={fac_arr.max():,.0f}"
    )
    logger.info(
        f"Min drainage: {MIN_WATERSHED_AREA_CELLS:,} cells "
        f"= ~{MIN_WATERSHED_AREA_CELLS * cell_area_km2:.1f} km2"
    )
    logger.info(f"Snap radius : {SNAP_DISTANCE} cells = {SNAP_DISTANCE * cell_w:.0f} m")

    # Step A: Find outlets
    logger.info("Finding stream outlets...")
    outlets = find_outlets(fac_arr, fdr_arr, fac_nd, fdr_nd,
                           MIN_WATERSHED_AREA_CELLS, logger)
    if not outlets:
        logger.error("No outlets found. Lower MIN_WATERSHED_AREA_CELLS.")
        sys.exit(1)

    # Step B: Snap
    logger.info("Snapping outlets to stream cells...")
    outlets = snap_outlets_to_stream(
        outlets, fac_arr, fac_nd, transform,
        SNAP_DISTANCE, MIN_WATERSHED_AREA_CELLS, logger
    )

    # Save pour points
    outlet_rows = [r for r, c in outlets]
    outlet_cols = [c for r, c in outlets]
    xs, ys = rasterio.transform.xy(transform, outlet_rows, outlet_cols)
    pour_gdf = gpd.GeoDataFrame(
        {
            "POUR_ID":   list(range(1, len(outlets) + 1)),
            "fac_value": [float(fac_arr[r, c]) for r, c in outlets],
        },
        geometry=gpd.points_from_xy(xs, ys),
        crs=fac_crs,
    )
    pour_shp = output_dir / "pourpoints_final.shp"
    pour_gdf.to_file(str(pour_shp))
    logger.info(f"  {len(outlets)} pour point(s) -> {pour_shp.name}")
    for _, row in pour_gdf.iterrows():
        logger.info(
            f"    POUR_ID={int(row.POUR_ID)}  FAC={row.fac_value:,.0f}  "
            f"({row.geometry.x:.1f}, {row.geometry.y:.1f})"
        )

    # Step C: Delineate watersheds (BFS)
    logger.info("Delineating watersheds (D8 BFS)...")
    with rasterio.open(str(fdr_path)) as src:
        fdr_meta = src.meta.copy()

    labels = _d8_watershed_bfs(fdr_arr, fdr_nd, outlet_rows, outlet_cols)
    labeled_cells = int((labels > 0).sum())
    logger.info(f"  BFS complete — {labeled_cells:,} cells assigned to {len(outlets)} watershed(s)")

    if labeled_cells == 0:
        logger.error("BFS returned no cells. Check outlet locations and FDR raster.")
        sys.exit(1)

    # Write raster (temporary)
    watersheds_tif = output_dir / "watersheds_tmp.tif"
    fdr_meta.update(dtype=rasterio.int32, nodata=0)
    with rasterio.open(str(watersheds_tif), "w", **fdr_meta) as dst:
        dst.write(labels, 1)

    # Vectorise
    logger.info("  Vectorising watersheds...")
    warnings.filterwarnings("ignore", message=".*Memory.*driver is deprecated.*")
    with rasterio.open(str(fdr_path)) as src:
        transform_v = src.transform
        crs_v = src.crs

    valid_mask = (labels != 0).astype(np.uint8)
    shapes_gen = rasterio.features.shapes(
        labels.astype(np.int32), mask=valid_mask, transform=transform_v
    )
    geoms, gridcodes = [], []
    for geom_dict, val in shapes_gen:
        geoms.append(shapely_shape(geom_dict))
        gridcodes.append(int(val))

    if not geoms:
        logger.error("Vectorisation produced no polygons.")
        sys.exit(1)

    gdf = gpd.GeoDataFrame({"gridcode": gridcodes}, geometry=geoms, crs=crs_v)
    gdf = gdf.dissolve(by="gridcode").reset_index()
    gdf["area_km2"] = gdf.geometry.area / 1e6

    watersheds_shp = output_dir / "watersheds.shp"
    gdf.to_file(str(watersheds_shp))
    logger.info(f"  Watersheds written: {watersheds_shp.name}")

    for _, row in gdf.iterrows():
        logger.info(
            f"    Watershed {int(row.gridcode)}: "
            f"{row.geometry.area:,.0f} m2  ({row.area_km2:.2f} km2)"
        )

    # Clean up temp raster
    watersheds_tif.unlink(missing_ok=True)

    return gdf, pour_gdf


# =============================================================================
# MAIN
# =============================================================================

def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("stream_and_watersheds.py — combined pipeline")
    logger.info("=" * 60)
    logger.info(f"DEM : {DEM_FILE}")
    logger.info(f"FAC : {FAC_FILE}")
    logger.info(f"FDR : {FDR_FILE}")
    logger.info(f"Out : {output_dir}")
    logger.info("-" * 60)

    start = time.time()

    # -----------------------------------------------------------------------
    # Part 1: Extract stream network
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=== PART 1: STREAM EXTRACTION ===")
    lines, transform, crs = run_stream_extraction(
        FAC_FILE, FDR_FILE, output_dir, logger
    )
    logger.info(f"Stream extraction complete — {len(lines):,} segments")

    # -----------------------------------------------------------------------
    # Part 2: Delineate watersheds
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=== PART 2: WATERSHED DELINEATION ===")
    gdf, pour_gdf = run_watershed_delineation(
        FAC_FILE, FDR_FILE, DEM_FILE, output_dir, logger
    )

    elapsed = time.time() - start
    logger.info("")
    logger.info("=" * 60)
    logger.info("ALL DONE")
    logger.info(f"  Stream network  : {output_dir / 'streams_connected.gpkg'}")
    if EXTRACT_LONGEST_BRANCH:
        logger.info(f"  Longest branch  : {output_dir / 'streams_longest_branch.gpkg'}")
    logger.info(f"  Pour points     : {output_dir / 'pourpoints_final.shp'}")
    logger.info(f"  Watersheds      : {output_dir / 'watersheds.shp'}")
    logger.info(f"  Watershed count : {len(gdf)}")
    logger.info(f"  Total time      : {elapsed / 60:.1f} min")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
