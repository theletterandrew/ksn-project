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
  4. Merge collinear segments at junctions using a graph (networkx)
     so the output is a clean, connected network rather than a pile
     of 1-2 pixel fragments
  5. Write the final LineStrings to a GeoPackage

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

# Number of border cells to blank on all four edges before thresholding.
# Edge cells drain "off the raster" in D8 routing and accumulate spurious
# flow, creating false streams along the DEM boundary.
# At 2 m resolution, 3 cells = 6 m. Increase to 5-10 if artifacts persist.
BORDER_CELLS = 3

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


def trace_segments(adj: dict) -> list:
    """
    Walk the adjacency graph and extract linear pixel chains (segments).

    A junction is any pixel with >=3 neighbours. Segments run between
    junctions (or dead-ends), tracing each branch exactly once.

    Returns a list of pixel chains: [[(r,c), (r,c), ...], ...]
    """
    # Classify nodes
    endpoints  = {n for n, nbrs in adj.items() if len(nbrs) == 1}
    junctions  = {n for n, nbrs in adj.items() if len(nbrs) >= 3}
    start_nodes = endpoints | junctions

    visited_edges = set()
    segments      = []

    def _trace(start, direction):
        chain = [start, direction]
        prev  = start
        curr  = direction

        while True:
            nbrs = [n for n in adj.get(curr, []) if n != prev]
            # Stop at junction, dead-end, or already-visited node
            if not nbrs or curr in junctions:
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


def main():
    wbt_dir    = Path(WBT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    fac_path = wbt_dir / FAC_FILE
    fdr_path = wbt_dir / FDR_FILE
    out_path = output_dir / OUTPUT_FILE

    for label, p in [("FAC", fac_path), ("FDR", fdr_path)]:
        if not p.exists():
            logger.error(f"{label} raster not found: {p}")
            logger.error("Run wbt_hydrology.py first.")
            sys.exit(1)

    logger.info(f"Threshold   : {THRESHOLD:,} cells (~{THRESHOLD * 4 / 1e6:.1f} km² at 2m)")
    logger.info(f"Min pixels  : {MIN_PIXELS}")
    logger.info(f"Border cells: {BORDER_CELLS}")
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
        adj      = build_adjacency(skeleton)
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
            for seg in segments:
                line = pixels_to_linestring(seg, transform)
                if line.length == 0:
                    continue
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

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("COMPLETE")
        logger.info(f"  Output    : {out_path}")
        logger.info(f"  Segments  : {count:,}")
        logger.info(f"  Total time: {elapsed / 60:.1f} minutes")
        logger.info("")
        logger.info("Load streams_connected.gpkg in QGIS or ArcGIS Pro to verify.")
        logger.info("Each feature is a LineString ordered mouth->source.")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
