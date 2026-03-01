"""
calculate_ksn.py
----------------
Calculates normalized channel steepness index (ksn) for watershed DEMs
using direct raster operations with numpy/scipy/rasterio.

Uses per-watershed DEM, FAC, and FDR rasters produced by clip_watersheds.py,
all on identical grids. No reprojection is performed anywhere in this script.

ksn = slope * (drainage_area ^ theta)   where theta = REFERENCE_CONCAVITY

Exports results as point shapefiles with ksn, slope, and area_km2 attributes.

USAGE:
    1. Run clip_watersheds.py first to produce per-watershed DEM, FAC, FDR files.
    2. Run:
       python calculate_ksn.py

Requirements:
    conda install -c conda-forge geopandas rasterio scipy numpy shapely
"""

import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import rasterio
import rasterio.transform
from scipy.ndimage import generic_filter
import geopandas as gpd
from shapely.geometry import Point

# Calculate the path to the project root (one level up from scripts/)
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

# =============================================================================
# CONFIG — Edit these before running
# =============================================================================

DEM_MOSAIC         = config.DATA_DEM_MOSAIC / "dem_mosaic.tif"  # Full mosaic — used to detect mosaic boundary edges
WATERSHED_DEMS_DIR = config.DATA_WATERSHEDS   # Watershed DEMs + FAC + FDR rasters
OUTPUT_DIR         = config.DATA_KSN          # Output ksn shapefiles

# Ksn calculation parameters
MIN_DRAINAGE_AREA_M2   = config.MIN_DRAINAGE_AREA_M2
REFERENCE_CONCAVITY    = config.REFERENCE_CONCAVITY
SMOOTHING_WINDOW       = config.SMOOTHING_WINDOW
SAMPLE_DISTANCE        = config.SAMPLE_DISTANCE
MIN_TRIBUTARY_LENGTH_M = config.MIN_TRIBUTARY_LENGTH_M

# =============================================================================
# END CONFIG
# =============================================================================


def setup_logging(output_dir: Path) -> logging.Logger:
    log_path = output_dir / "calculate_ksn.log"
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


def calculate_gradient_smoothed(
    dem: np.ndarray, nodata, cellsize: float, window_size: int = 5
) -> np.ndarray:
    """
    Calculate slope (m/m) using a nodata-aware smoothed gradient.

    Masks nodata cells to NaN before smoothing so that nodata areas outside
    the watershed boundary do not bleed into the kernel and corrupt slopes
    at the watershed edge.
    """
    dem_masked = dem.astype(np.float32)
    if nodata is not None:
        dem_masked[dem == nodata] = np.nan

    # Suppress scipy's "Mean of empty slice" warning — expected at nodata
    # boundary pixels where the entire kernel window is NaN.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        smoothed = generic_filter(
            dem_masked, np.nanmean, size=window_size, mode="nearest"
        )

    dy, dx = np.gradient(smoothed, cellsize)
    return np.sqrt(dx**2 + dy**2)


def _build_upstream_index(
    fdr_arr: np.ndarray, fdr_nodata, stream_mask: np.ndarray
) -> dict:
    """
    Build a dict mapping each stream cell (r, c) to its upstream stream
    neighbours using the WBT ESRI D8 pointer convention.
    """
    D8 = {
        ( 0,  1): 1,
        ( 1,  1): 2,
        ( 1,  0): 4,
        ( 1, -1): 8,
        ( 0, -1): 16,
        (-1, -1): 32,
        (-1,  0): 64,
        (-1,  1): 128,
    }
    UPSTREAM_FDR = {
        (dr, dc): D8[(-dr, -dc)]
        for (dr, dc) in D8
        if (-dr, -dc) in D8
    }

    nrows, ncols = fdr_arr.shape
    stream_set   = set(zip(*np.where(stream_mask)))
    upstream     = {}

    for r, c in stream_set:
        ups = []
        for (dr, dc), expected_fdr in UPSTREAM_FDR.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < nrows and 0 <= nc < ncols):
                continue
            if (nr, nc) not in stream_set:
                continue
            cell_fdr = int(fdr_arr[nr, nc])
            if fdr_nodata is not None and cell_fdr == int(fdr_nodata):
                continue
            if cell_fdr == expected_fdr:
                ups.append((nr, nc))
        upstream[(r, c)] = ups

    return upstream


def _trace_channel_ordered(
    outlet_rc: tuple,
    upstream: dict,
    sample_dist_m: float,
    cellsize: float,
) -> list:
    """
    BFS from outlet_rc upward, collecting one sample point per
    ~sample_dist_m of channel distance.

    Returns a list of (row, col) tuples ordered outlet -> headwater.
    """
    from collections import deque

    sample_step = max(1, int(sample_dist_m / cellsize))
    visited     = set()
    result      = []

    queue = deque([(outlet_rc, 0)])

    while queue:
        cell, steps = queue.popleft()
        if cell in visited:
            continue
        visited.add(cell)

        steps += 1
        if steps >= sample_step or cell == outlet_rc:
            result.append(cell)
            steps = 0

        for up_cell in upstream.get(cell, []):
            if up_cell not in visited:
                queue.append((up_cell, steps))

    return result


def _filter_short_tributaries(
    sampled_cells: list,
    upstream: dict,
    cellsize: float,
    min_length_m: float,
    outlet_rc: tuple,
) -> list:
    """
    Remove sampled points on dangling tributary tips shorter than min_length_m.

    Headwaters are identified within the SAMPLED network only — a cell is a
    sampled headwater if none of its upstream neighbours are also in sampled_cells.
    This handles the case where the full stream mask has cells above every sampled
    point (so nothing appears as a headwater in the full upstream dict).

    For each sampled headwater we walk downstream through sampled_cells until we
    reach a sampled junction (2+ sampled upstream neighbours) or the end of the
    sampled network. The branch is removed if:
      - its length is below min_length_m, AND
      - it does not terminate at outlet_rc (which would mean it IS the main stem)
    """
    if min_length_m <= 0:
        return sampled_cells

    sampled_set = set(sampled_cells)
    min_steps   = max(1, int(min_length_m / cellsize))

    # Build sampled-only upstream and downstream indices
    sampled_upstream = {
        c: [u for u in upstream.get(c, []) if u in sampled_set]
        for c in sampled_set
    }
    sampled_downstream = {}
    for cell, ups in sampled_upstream.items():
        for up in ups:
            sampled_downstream[up] = cell

    # Headwaters in sampled network: no sampled upstream neighbours
    headwaters = [c for c in sampled_set if not sampled_upstream.get(c)]

    cells_to_remove = set()

    for tip in headwaters:
        branch  = [tip]
        current = tip
        steps   = 0

        while True:
            ds = sampled_downstream.get(current)
            if ds is None:
                # End of sampled network reached.
                # If current IS the outlet this branch is the main stem — keep.
                # Otherwise it's an orphaned stub with no downstream — remove.
                if current == outlet_rc:
                    branch = []
                break
            steps += 1
            if len(sampled_upstream.get(ds, [])) > 1:
                # Reached a sampled junction — stop here
                break
            branch.append(ds)
            current = ds

        if branch and steps < min_steps:
            cells_to_remove.update(branch)

    return [c for c in sampled_cells if c not in cells_to_remove]

def extract_stream_points(
    dem_path: Path,
    fac_path: Path,
    fdr_path: Path,
    min_area_m2: float,
    sample_dist: float,
    theta: float,
    window_size: int,
    logger: logging.Logger,
) -> gpd.GeoDataFrame:
    """
    Extract ksn sample points for a single watershed.

    All three rasters (dem, fac, fdr) must be pre-clipped by
    clip_watersheds.py and share an identical grid. No reprojection
    is performed — a grid mismatch raises immediately.
    """
    # ------------------------------------------------------------------
    # 1. Load watershed DEM
    # ------------------------------------------------------------------
    with rasterio.open(str(dem_path)) as src:
        dem       = src.read(1)
        transform = src.transform
        crs       = src.crs
        cellsize  = src.res[0]
        nodata    = src.nodata

    # ------------------------------------------------------------------
    # 2. Load per-watershed FAC — must be grid-aligned to DEM
    # ------------------------------------------------------------------
    with rasterio.open(str(fac_path)) as fac_src:
        if fac_src.transform != transform or fac_src.shape != dem.shape:
            raise RuntimeError(
                f"Pre-clipped FAC grid does not match DEM grid. "
                f"FAC: {fac_src.shape} | DEM: {dem.shape}. "
                f"Re-run clip_watersheds.py."
            )
        fac        = fac_src.read(1).astype(np.float32)
        fac_nodata = fac_src.nodata
        if fac_nodata is not None:
            fac[fac == fac_nodata] = 0.0

    area_m2 = fac * (cellsize ** 2)

    logger.info(
        f"  Max area in DEM: {np.nanmax(area_m2):.2f} m²  "
        f"| Threshold: {min_area_m2:.2f} m²"
    )

    # ------------------------------------------------------------------
    # 3. Load per-watershed FDR — must be grid-aligned to DEM.
    #    Loaded before stream_mask so we can use mosaic-boundary
    #    detection to blank spurious edge cells from the stream mask.
    # ------------------------------------------------------------------
    with rasterio.open(str(fdr_path)) as fdr_src:
        if fdr_src.transform != transform or fdr_src.shape != dem.shape:
            raise RuntimeError(
                f"Pre-clipped FDR grid does not match DEM grid. "
                f"FDR: {fdr_src.shape} | DEM: {dem.shape}. "
                f"Re-run clip_watersheds.py."
            )
        fdr_raw    = fdr_src.read(1).astype(np.int32)
        fdr_nodata = fdr_src.nodata

    # ------------------------------------------------------------------
    # 4. Build stream mask — FAC threshold AND valid DEM cells only.
    #    Masking against valid DEM pixels eliminates spurious stream
    #    points from FAC bleed at the watershed boundary.
    # ------------------------------------------------------------------
    if nodata is not None:
        dem_valid = dem != nodata
    else:
        dem_valid = np.isfinite(dem.astype(np.float32))

    stream_mask = (area_m2 >= min_area_m2) & dem_valid

    if not stream_mask.any():
        logger.warning("  No stream cells above threshold")
        return None

    # ------------------------------------------------------------------
    # 5. Compute slope with nodata-aware smoothing
    # ------------------------------------------------------------------
    logger.info("  Computing slope...")
    slope = calculate_gradient_smoothed(dem, nodata, cellsize, window_size)

    # ------------------------------------------------------------------
    # 6. Compute ksn = slope * area^theta
    # ------------------------------------------------------------------
    logger.info("  Computing ksn...")
    area_safe = np.maximum(area_m2, 1.0)
    ksn = slope * (area_safe ** theta)
    ksn = np.where(np.isfinite(ksn), ksn, 0.0)

    # ------------------------------------------------------------------
    # 7. Filter invalid pixels before sampling
    # ------------------------------------------------------------------
    stream_rows, stream_cols = np.where(stream_mask)
    valid = (
        (nodata is None or dem[stream_rows, stream_cols] != nodata)
        & np.isfinite(slope[stream_rows, stream_cols])
        & np.isfinite(ksn[stream_rows, stream_cols])
        & (ksn[stream_rows, stream_cols] > 0)
    )
    stream_rows = stream_rows[valid]
    stream_cols = stream_cols[valid]

    if len(stream_rows) == 0:
        logger.warning("  No valid stream cells after filtering")
        return None

    valid_stream_mask = np.zeros(dem.shape, dtype=bool)
    valid_stream_mask[stream_rows, stream_cols] = True

    # ------------------------------------------------------------------
    # 8. Sample along D8 flow paths (outlet -> headwater)
    # ------------------------------------------------------------------
    logger.info("  Tracing channel paths for ordered sampling...")
    upstream = _build_upstream_index(fdr_raw, fdr_nodata, valid_stream_mask)

    fac_at_stream = area_m2[stream_rows, stream_cols]
    outlet_idx    = int(np.argmax(fac_at_stream))
    outlet_rc     = (int(stream_rows[outlet_idx]), int(stream_cols[outlet_idx]))

    sampled_cells = _trace_channel_ordered(
        outlet_rc, upstream, sample_dist, cellsize
    )

    if not sampled_cells:
        logger.warning("  Channel tracing produced no sample points")
        return None

    logger.info(f"  {len(sampled_cells)} sample points along channel paths")

    # ------------------------------------------------------------------
    # 9. Remove dangling tributary tips shorter than MIN_TRIBUTARY_LENGTH_M
    # ------------------------------------------------------------------
    if MIN_TRIBUTARY_LENGTH_M > 0:
        before = len(sampled_cells)
        sampled_cells = _filter_short_tributaries(
            sampled_cells, upstream, cellsize, MIN_TRIBUTARY_LENGTH_M, outlet_rc
        )
        dropped_tribs = before - len(sampled_cells)
        if dropped_tribs:
            logger.info(
                f"  Dropped {dropped_tribs} points on short tributaries "
                f"(< {MIN_TRIBUTARY_LENGTH_M:.0f} m)"
            )
    # DIAGNOSTIC
    headwaters_debug = [c for c in set(sampled_cells) if not upstream.get(c)]
    logger.info(f"  Headwater sampled cells: {len(headwaters_debug)}")
    for tip in headwaters_debug[:5]:
        downstream_debug = {}
        for cell, ups in upstream.items():
            for up in ups:
                downstream_debug[up] = cell
        current = tip
        steps = 0
        while True:
            ds = downstream_debug.get(current)
            if ds is None:
                break
            steps += 1
            if len(upstream.get(ds, [])) > 1:
                break
            if ds not in set(sampled_cells):
                break
            current = ds
        logger.info(f"    tip={tip} steps_to_junction={steps} min_steps={int(MIN_TRIBUTARY_LENGTH_M / cellsize)}")
    # END DIAGNOSTIC
    
    if not sampled_cells:
        logger.warning("  All points removed by tributary length filter")
        return None

    # ------------------------------------------------------------------
    # 10. Remove points within 3 cells of the mosaic boundary.
    #
    #    WBT D8 routing produces spurious high-accumulation values along
    #    the outermost rows/cols of the mosaic. Rather than blanking the
    #    stream_mask before tracing (which kills watershed_4's outlet),
    #    we let the BFS run freely and then drop any sampled point whose
    #    geographic coordinates fall within BORDER_CELLS of the mosaic edge.
    # ------------------------------------------------------------------
    try:
        with rasterio.open(str(DEM_MOSAIC)) as mosaic_src:
            mb       = mosaic_src.bounds
            border_m = 3 * mosaic_src.res[0]   # 3 cells in map units

        before = len(sampled_cells)
        sampled_cells = [
            (r, c) for r, c in sampled_cells
            if not (
                rasterio.transform.xy(transform, r, c)[0] <= mb.left   + border_m or
                rasterio.transform.xy(transform, r, c)[0] >= mb.right  - border_m or
                rasterio.transform.xy(transform, r, c)[1] <= mb.bottom + border_m or
                rasterio.transform.xy(transform, r, c)[1] >= mb.top    - border_m
            )
        ]
        dropped = before - len(sampled_cells)
        if dropped:
            logger.info(f"  Dropped {dropped} points within {3}-cell mosaic border")
    except Exception as e:
        logger.warning(f"  Could not filter mosaic border points: {e}")

    if not sampled_cells:
        logger.warning("  All points were within mosaic border — skipping")
        return None

    # ------------------------------------------------------------------
    # 11. Build output GeoDataFrame
    # ------------------------------------------------------------------
    s_rows = [rc[0] for rc in sampled_cells]
    s_cols = [rc[1] for rc in sampled_cells]
    xs, ys = rasterio.transform.xy(transform, s_rows, s_cols)

    return gpd.GeoDataFrame(
        {
            "ksn":      [float(ksn[r, c])           for r, c in sampled_cells],
            "slope":    [float(slope[r, c])          for r, c in sampled_cells],
            "area_km2": [float(area_m2[r, c]) / 1e6 for r, c in sampled_cells],
        },
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs=crs,
    )


def calculate_ksn_for_watershed(
    dem_path: Path,
    fac_path: Path,
    fdr_path: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> tuple[bool, str]:
    """
    Calculates ksn for a single watershed and exports to shapefile.
    Returns (success, output_path).
    """
    watershed_id = dem_path.stem
    out_shp      = output_dir / f"{watershed_id}_ksn.shp"

    if out_shp.exists():
        return (True, str(out_shp))

    try:
        logger.info("  Extracting stream points...")
        gdf = extract_stream_points(
            dem_path, fac_path, fdr_path,
            MIN_DRAINAGE_AREA_M2, SAMPLE_DISTANCE,
            REFERENCE_CONCAVITY, SMOOTHING_WINDOW,
            logger,
        )

        if gdf is None or len(gdf) == 0:
            logger.warning("  No streams found — skipping")
            return (False, "")

        logger.info("  Exporting to shapefile...")
        gdf.to_file(str(out_shp))

        logger.info(
            f"  Exported {len(gdf)} points  |  "
            f"ksn mean: {gdf['ksn'].mean():.1f}  |  "
            f"ksn std: {gdf['ksn'].std():.1f}"
        )
        return (True, str(out_shp))

    except Exception as e:
        logger.error(f"  Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
            f = out_shp.parent / (out_shp.stem + ext)
            if f.exists():
                try:
                    f.unlink()
                except Exception:
                    pass
        return (False, "")


def main():
    dems_dir   = Path(WATERSHED_DEMS_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    # Collect watershed DEMs, excluding _fac and _fdr clips
    dem_files = sorted(
        f for f in dems_dir.glob("watershed_*.tif")
        if not f.stem.endswith("_fac") and not f.stem.endswith("_fdr")
    )

    if not dem_files:
        logger.error(f"No watershed_*.tif files found in: {dems_dir}")
        logger.error("Run clip_watersheds.py first.")
        sys.exit(1)

    # Verify matching FAC and FDR files exist before starting
    missing = [
        f for f in dem_files
        if not (dems_dir / f"{f.stem}_fac.tif").exists()
        or not (dems_dir / f"{f.stem}_fdr.tif").exists()
    ]
    if missing:
        logger.error(
            f"{len(missing)} watershed(s) missing a _fac.tif or _fdr.tif: "
            f"{[f.stem for f in missing]}. Re-run clip_watersheds.py."
        )
        sys.exit(1)

    total = len(dem_files)
    logger.info(f"Found {total} watershed DEMs")
    logger.info(f"Input DEMs           : {dems_dir}")
    logger.info(f"Output dir           : {output_dir}")
    logger.info(f"Min drainage area    : {MIN_DRAINAGE_AREA_M2 / 1e6:.3f} km²")
    logger.info(f"Reference concavity  : {REFERENCE_CONCAVITY}")
    logger.info(f"Smoothing window     : {SMOOTHING_WINDOW} cells")
    logger.info(f"Sample distance      : {SAMPLE_DISTANCE} m")
    logger.info("-" * 60)

    start_time = time.time()
    succeeded  = 0
    failed     = 0
    skipped    = 0

    for i, dem_path in enumerate(dem_files, start=1):
        watershed_id = dem_path.stem
        fac_path     = dems_dir / f"{watershed_id}_fac.tif"
        fdr_path     = dems_dir / f"{watershed_id}_fdr.tif"
        out_shp      = output_dir / f"{watershed_id}_ksn.shp"

        if out_shp.exists():
            skipped += 1
            logger.info(f"[{i:3d}/{total}] SKIP  {watershed_id} — already exists")
            continue

        logger.info(f"[{i:3d}/{total}] START {watershed_id}")
        tile_start = time.time()

        success, _ = calculate_ksn_for_watershed(
            dem_path, fac_path, fdr_path, output_dir, logger
        )

        if success:
            succeeded += 1
            tile_time = time.time() - tile_start
            elapsed   = time.time() - start_time
            rate      = i / elapsed
            eta_min   = (total - i) / rate / 60 if rate > 0 else 0
            logger.info(
                f"[{i:3d}/{total}] OK    {watershed_id}  |  "
                f"{tile_time:.1f}s  |  ETA {eta_min:.1f} min"
            )
        else:
            failed += 1
            logger.error(f"[{i:3d}/{total}] FAIL  {watershed_id}")

    elapsed_total = time.time() - start_time
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info(f"  Total watersheds : {total}")
    logger.info(f"  Succeeded        : {succeeded}")
    logger.info(f"  Skipped          : {skipped}")
    logger.info(f"  Failed           : {failed}")
    logger.info(f"  Output dir       : {output_dir}")
    logger.info(f"  Total time       : {elapsed_total / 60:.1f} minutes")
    logger.info("")
    logger.info("Load *_ksn.shp files in ArcGIS Pro or QGIS to visualize ksn values.")


if __name__ == "__main__":
    main()
