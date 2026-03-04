"""
delineate_watersheds.py
-----------------------
Identifies the true mouth (most-downstream cell) of every stream that exits
the study area, then delineates the contributing watershed for each using the
WhiteboxTools flow direction raster.

Requires only the three rasters produced by wbt_hydrology.py:
    - dem_filled.tif        (filled DEM — used only for validation)
    - flow_direction.tif    (WBT D8 pointer raster)
    - flow_accumulation.tif (WBT flow accumulation)

No stream vector layer is needed.

ALGORITHM
---------
1.  Build a stream mask: all FAC cells >= MIN_DRAINAGE_AREA_CELLS.
2.  Find raw outlet cells: stream cells whose single D8 downstream neighbour
    is not also a stream cell (outside raster, nodata, or below threshold).
3.  Deduplicate to one outlet per exiting stream: for each raw outlet, walk
    downstream through the full raster; if the path reaches another raw outlet
    before leaving the grid, the upstream one is redundant and is discarded.
4.  Snap each kept outlet to the highest-FAC stream cell within SNAP_DISTANCE,
    constrained to stay on the same stream (FAC cap prevents cross-tributary
    jumps that WBT SnapPourPoints causes).
5.  Delineate watersheds with a pure-numpy D8 BFS and vectorise.

USAGE
-----
    conda activate arcgispro-py3
    python delineate_watersheds.py

Requirements: geopandas, rasterio, numpy, WhiteboxTools v2.4.0+
"""

import logging
import sys
import time
import warnings
from pathlib import Path
import subprocess

import numpy as np
import rasterio
import rasterio.features
import rasterio.transform
import geopandas as gpd
from shapely.geometry import shape as shapely_shape

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

# =============================================================================
# CONFIG
# =============================================================================

WBT_DIR    = config.DATA_SCRATCH_WBT
OUTPUT_DIR = config.DATA_SCRATCH_WATERSHEDS
WBT_EXE    = config.WBT_EXE

FILLED_DEM = WBT_DIR / "dem_filled.tif"
FDR_FILE   = WBT_DIR / "flow_direction.tif"
FAC_FILE   = WBT_DIR / "flow_accumulation.tif"

MIN_DRAINAGE_AREA_CELLS = config.MIN_WATERSHED_AREA
SNAP_DISTANCE           = config.SNAP_DISTANCE   # cells

DEBUG_POUR_POINTS = False

# =============================================================================
# END CONFIG
# =============================================================================

# WBT D8 pointer values -> (row_offset, col_offset)
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


def setup_logging(output_dir: Path) -> logging.Logger:
    log_path = output_dir / "delineate_watersheds.log"
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


def run_wbt(tool: str, args: dict, logger: logging.Logger, timeout: int = 600) -> bool:
    """Run a WhiteboxTools command, streaming stdout/stderr in real time."""
    import threading
    cmd = [str(WBT_EXE), f"--run={tool}"] + [f"--{k}={v}" for k, v in args.items()]
    logger.info(f"  WBT {tool}: {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        def _drain(pipe, fn):
            for line in iter(pipe.readline, ""):
                line = line.rstrip()
                if line:
                    fn(f"    WBT: {line}")
            pipe.close()

        for pipe, fn in [(proc.stdout, logger.info), (proc.stderr, logger.warning)]:
            threading.Thread(target=_drain, args=(pipe, fn), daemon=True).start()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            logger.error(f"  {tool} killed after {timeout}s — check inputs.")
            return False

        if proc.returncode != 0:
            logger.error(f"  {tool} failed (rc={proc.returncode})")
            return False
        return True
    except Exception as e:
        logger.error(f"  Failed to run {tool}: {e}")
        return False


def find_outlets(
    fac_arr: np.ndarray,
    fdr_arr: np.ndarray,
    fac_nd,
    fdr_nd,
    min_accum_cells: int,
    logger: logging.Logger,
) -> list:
    """
    Return a list of (row, col) outlet cells — one per stream that exits the
    study area.

    Step A — Raw outlets
    --------------------
    A stream cell is a raw outlet when its D8 downstream neighbour is not
    also a stream cell.  "Not a stream cell" means the neighbour is:
      - outside the raster extent, OR
      - a nodata cell in the FDR raster, OR
      - below the FAC threshold (flow leaves the qualifying network)
    FDR==0 (WBT unroutable / edge cell) also qualifies.

    Step B — Deduplicate to one outlet per trunk
    --------------------------------------------
    A study area with several drainages exiting at different points will
    correctly produce one raw outlet per exit.  But if a stream runs
    parallel to the raster edge for many cells before exiting, multiple
    adjacent cells can all be raw outlets on the same trunk.  We discard
    any raw outlet that is upstream of another raw outlet on the same D8
    path — keep only the most-downstream one.
    """
    nrows, ncols = fac_arr.shape
    stream_mask  = fac_arr >= min_accum_cells

    stream_rows, stream_cols = np.where(stream_mask)
    if len(stream_rows) == 0:
        logger.error(
            f"No FAC cells >= {min_accum_cells:,}. Check MIN_WATERSHED_AREA."
        )
        return []
    logger.info(f"  {len(stream_rows):,} stream cells at FAC >= {min_accum_cells:,}")

    # ------------------------------------------------------------------
    # Step A: raw outlets
    # ------------------------------------------------------------------
    def _next_cell(r, c):
        """Return (nr, nc) of the D8 downstream neighbour, or None if off-grid/nodata."""
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

    raw_outlets = set()
    for r, c in zip(stream_rows.tolist(), stream_cols.tolist()):
        nxt = _next_cell(r, c)
        if nxt is None or not stream_mask[nxt[0], nxt[1]]:
            raw_outlets.add((r, c))

    logger.info(f"  {len(raw_outlets)} raw outlet cell(s) before deduplication")

    if not raw_outlets:
        logger.warning(
            "  No raw outlets found — basin may be internally draining.\n"
            "  Falling back to the single highest-FAC stream cell."
        )
        best     = int(np.argmax(fac_arr * stream_mask))
        return [(int(best // ncols), int(best % ncols))]

    # ------------------------------------------------------------------
    # Step B: remove upstream duplicates.
    #
    # Walk downstream from each raw outlet (following D8 through the full
    # raster, not just stream cells).  If the walk reaches another raw
    # outlet before leaving the raster or hitting nodata, the starting
    # outlet is upstream of it and is therefore redundant.
    # ------------------------------------------------------------------
    redundant = set()

    for (r0, c0) in raw_outlets:
        r, c = r0, c0
        for _ in range(nrows * ncols):
            nxt = _next_cell(r, c)
            if nxt is None:
                break   # left the raster / hit nodata / hit FDR=0 — keep (r0,c0)
            r, c = nxt
            if (r, c) in raw_outlets:
                redundant.add((r0, c0))
                break

    kept = [rc for rc in raw_outlets if rc not in redundant]
    logger.info(
        f"  {len(redundant)} upstream duplicate(s) removed — "
        f"{len(kept)} outlet(s) kept"
    )
    return kept


def snap_outlets_to_stream(
    outlets: list,
    fac_arr: np.ndarray,
    fac_nd,
    transform,
    snap_cells: int,
    min_accum_cells: int,
    logger: logging.Logger,
) -> list:
    """
    For each outlet cell, search within snap_cells radius for the stream cell
    with the highest FAC that belongs to the same drainage.

    The FAC cap (2x the outlet's own FAC) prevents the search from jumping
    to an adjacent higher-order trunk stream.

    Returns a list of (row, col) snapped outlet cells (same length as input).
    """
    nrows, ncols = fac_arr.shape
    stream_mask  = fac_arr >= min_accum_cells
    snapped      = []

    for (r0, c0) in outlets:
        orig_fac = float(fac_arr[r0, c0])
        cap_fac  = orig_fac * 2.0

        best_r, best_c, best_fac = r0, c0, orig_fac

        for dr in range(-snap_cells, snap_cells + 1):
            for dc in range(-snap_cells, snap_cells + 1):
                nr, nc = r0 + dr, c0 + dc
                if not (0 <= nr < nrows and 0 <= nc < ncols):
                    continue
                if not stream_mask[nr, nc]:
                    continue
                fval = float(fac_arr[nr, nc])
                if fval > cap_fac:
                    continue   # different stream — skip
                if fval > best_fac:
                    best_r, best_c, best_fac = nr, nc, fval

        if (best_r, best_c) != (r0, c0):
            ox, oy = rasterio.transform.xy(transform, r0, c0)
            sx, sy = rasterio.transform.xy(transform, best_r, best_c)
            dist   = ((sx - ox) ** 2 + (sy - oy) ** 2) ** 0.5
            logger.info(
                f"  Snapped ({r0},{c0}) FAC={orig_fac:,.0f} -> "
                f"({best_r},{best_c}) FAC={best_fac:,.0f}  dist={dist:.0f} m"
            )
        snapped.append((best_r, best_c))

    return snapped


def _d8_watershed_numpy(
    fdr_arr: np.ndarray,
    fdr_nd,
    outlet_rows: list,
    outlet_cols: list,
) -> np.ndarray:
    """
    BFS upstream watershed delineation.
    Returns int32 label array: 0 = unassigned, 1..N = watershed ID.
    """
    from collections import deque

    nrows, ncols = fdr_arr.shape
    D8_REVERSE = {
        ( 0,  1): 1,   ( 1,  1): 2,   ( 1,  0): 4,   ( 1, -1): 8,
        ( 0, -1): 16,  (-1, -1): 32,  (-1,  0): 64,  (-1,  1): 128,
    }
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


def delineate_and_vectorise(
    fdr_path: Path,
    outlet_rows: list,
    outlet_cols: list,
    watersheds_tif: Path,
    watersheds_shp: Path,
    logger: logging.Logger,
) -> gpd.GeoDataFrame:
    """Run BFS delineation, write raster + shapefile, return GeoDataFrame."""
    with rasterio.open(fdr_path) as src:
        fdr_arr   = src.read(1)
        fdr_nd    = src.nodata
        crs       = src.crs
        transform = src.transform
        fdr_meta  = src.meta.copy()

    logger.info(
        f"  D8 BFS: {len(outlet_rows)} outlet(s), "
        f"{fdr_arr.shape[1]}x{fdr_arr.shape[0]} grid"
    )
    labels  = _d8_watershed_numpy(fdr_arr, fdr_nd, outlet_rows, outlet_cols)
    labeled = int((labels > 0).sum())
    logger.info(f"  BFS complete — {labeled:,} cells assigned")

    if labeled == 0:
        logger.error(
            "BFS returned no cells. Check outlet locations and FDR raster."
        )
        sys.exit(1)

    fdr_meta.update(dtype=rasterio.int32, nodata=0)
    with rasterio.open(watersheds_tif, "w", **fdr_meta) as dst:
        dst.write(labels, 1)

    logger.info("  Vectorising...")
    warnings.filterwarnings(
        "ignore", message=".*Memory.*driver is deprecated.*", category=RuntimeWarning
    )
    valid_mask = (labels != 0).astype(np.uint8)
    shapes_gen = rasterio.features.shapes(
        labels.astype(np.int32), mask=valid_mask, transform=transform
    )
    geoms, gridcodes = [], []
    for geom_dict, val in shapes_gen:
        geoms.append(shapely_shape(geom_dict))
        gridcodes.append(int(val))

    if not geoms:
        logger.error("Vectorisation produced no polygons.")
        sys.exit(1)

    gdf = gpd.GeoDataFrame({"gridcode": gridcodes}, geometry=geoms, crs=crs)
    gdf = gdf.dissolve(by="gridcode").reset_index()
    gdf["area_km2"] = gdf.geometry.area / 1e6
    gdf.to_file(watersheds_shp)
    return gdf


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    fdr_path = Path(FDR_FILE)
    fac_path = Path(FAC_FILE)
    dem_path = Path(FILLED_DEM)

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    missing = [
        f"  {lbl}: {p}"
        for lbl, p in [
            ("Flow direction (FDR)", fdr_path),
            ("Flow accumulation (FAC)", fac_path),
            ("Filled DEM", dem_path),
        ]
        if not p.exists()
    ]
    if missing:
        for m in missing:
            logger.error(m)
        logger.error("Run wbt_hydrology.py first.")
        sys.exit(1)

    with rasterio.open(fac_path) as src:
        cell_w    = abs(src.res[0])
        fac_shape = src.shape
        transform = src.transform
        fac_crs   = src.crs
        fac_arr   = src.read(1).astype(np.float64)
        fac_nd    = src.nodata
        if fac_nd is not None:
            fac_arr[fac_arr == fac_nd] = 0.0

    with rasterio.open(fdr_path) as src:
        fdr_arr = src.read(1)
        fdr_nd  = src.nodata

    cell_area_km2 = (cell_w ** 2) / 1e6
    logger.info(
        f"FAC raster  : {fac_shape[1]}x{fac_shape[0]} px @ {cell_w:.1f} m  "
        f"max={fac_arr.max():,.0f}"
    )
    logger.info(
        f"Min drainage: {MIN_DRAINAGE_AREA_CELLS:,} cells "
        f"= ~{MIN_DRAINAGE_AREA_CELLS * cell_area_km2:.1f} km2"
    )
    logger.info(f"Snap radius : {SNAP_DISTANCE} cells = {SNAP_DISTANCE * cell_w:.0f} m")
    logger.info("-" * 60)

    start_time = time.time()

    # ------------------------------------------------------------------
    # Step 1: Find outlets
    # ------------------------------------------------------------------
    logger.info("Step 1: Finding stream outlets...")
    outlets = find_outlets(
        fac_arr, fdr_arr, fac_nd, fdr_nd, MIN_DRAINAGE_AREA_CELLS, logger
    )
    if not outlets:
        logger.error("No outlets found. Lower MIN_WATERSHED_AREA in config.py.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Snap to highest-FAC stream cell within snap radius
    # ------------------------------------------------------------------
    logger.info("Step 2: Snapping outlets to stream cells...")
    outlets = snap_outlets_to_stream(
        outlets, fac_arr, fac_nd, transform,
        SNAP_DISTANCE, MIN_DRAINAGE_AREA_CELLS, logger
    )

    # Save pour points
    outlet_rows = [r for r, c in outlets]
    outlet_cols = [c for r, c in outlets]
    xs, ys = rasterio.transform.xy(transform, outlet_rows, outlet_cols)

    pour_gdf = gpd.GeoDataFrame(
        {
            "POUR_ID":   range(1, len(outlets) + 1),
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

    if DEBUG_POUR_POINTS:
        for r, c in outlets:
            logger.info(f"  DEBUG ({r},{c}) FDR={fdr_arr[r,c]}  FAC={fac_arr[r,c]:,.0f}")
            logger.info(f"  7x7 FDR neighbourhood:\n{fdr_arr[max(0,r-3):r+4, max(0,c-3):c+4]}")

    # ------------------------------------------------------------------
    # Step 3: Delineate watersheds
    # ------------------------------------------------------------------
    logger.info("Step 3: Delineating watersheds...")
    watersheds_tif = output_dir / "watersheds.tif"
    watersheds_shp = output_dir / "watersheds.shp"

    gdf = delineate_and_vectorise(
        fdr_path, outlet_rows, outlet_cols,
        watersheds_tif, watersheds_shp, logger
    )

    # ------------------------------------------------------------------
    # Step 4: Area statistics
    # ------------------------------------------------------------------
    logger.info("Step 4: Watershed area statistics:")
    for _, row in gdf.iterrows():
        logger.info(
            f"  Watershed {int(row.gridcode)}: "
            f"{row.geometry.area:,.0f} m2  ({row.area_km2:.2f} km2)"
        )

    watersheds_tif.unlink(missing_ok=True)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info(f"  Watersheds : {watersheds_shp}")
    logger.info(f"  Pour points: {pour_shp}")
    logger.info(f"  Count      : {len(gdf)}")
    logger.info(f"  Time       : {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
