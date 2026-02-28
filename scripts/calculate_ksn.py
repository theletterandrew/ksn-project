"""
calculate_ksn.py
----------------
Calculates normalized channel steepness index (ksn) for watershed DEMs
using direct raster operations with numpy/scipy/rasterio.

Uses per-watershed FAC rasters produced by clip_watersheds.py (which are
grid-aligned to their DEM), eliminating the reprojection that caused
spurious stream points in previous versions.

ksn = slope * (drainage_area ^ theta)   where theta = REFERENCE_CONCAVITY

Exports results as point shapefiles with ksn, slope, and area_km2 attributes.

USAGE:
    1. Run clip_watersheds.py first to produce per-watershed DEM and FAC files.
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
from rasterio.warp import reproject, Resampling
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

WBT_DIR            = config.DATA_SCRATCH_WBT    # WBT outputs (full FDR)
WATERSHED_DEMS_DIR = config.DATA_WATERSHEDS      # Watershed DEMs + per-watershed FAC rasters
OUTPUT_DIR         = config.DATA_KSN             # Output ksn shapefiles

# Full-mosaic FDR — still used for upstream tracing (reprojected to each
# watershed grid as needed; FDR reproject is acceptable since we only need
# the pointer direction, not an exact accumulation value).
FDR_FILE = "flow_direction.tif"

# Ksn calculation parameters
MIN_DRAINAGE_AREA_M2 = config.MIN_DRAINAGE_AREA_M2
REFERENCE_CONCAVITY  = config.REFERENCE_CONCAVITY
SMOOTHING_WINDOW     = config.SMOOTHING_WINDOW
SAMPLE_DISTANCE      = config.SAMPLE_DISTANCE

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

    fac_path must be a per-watershed FAC raster produced by
    clip_watersheds.py, grid-aligned to dem_path. No reprojection
    of the FAC is performed — a grid mismatch raises immediately.
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
    # 2. Load per-watershed FAC — must be grid-aligned to DEM.
    #    clip_watersheds.py guarantees this; raise clearly if violated.
    # ------------------------------------------------------------------
    with rasterio.open(str(fac_path)) as fac_src:
        if fac_src.transform != transform or fac_src.shape != dem.shape:
            raise RuntimeError(
                f"Pre-clipped FAC grid does not match DEM grid. "
                f"FAC: {fac_src.shape} | DEM: {dem.shape}. "
                f"Re-run clip_watersheds.py to regenerate the FAC clip."
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
    # 3. Build stream mask — FAC threshold AND valid DEM cells only.
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
    
    # DIAGNOSTIC — remove after diagnosis
    logger.info(f"  dem shape: {dem.shape}, nodata: {nodata}")
    logger.info(f"  dem nodata cells: {(dem == nodata).sum() if nodata is not None else 0:,}")
    logger.info(f"  dem_valid cells: {dem_valid.sum():,}")
    logger.info(f"  fac > 0 cells: {(fac > 0).sum():,}")
    logger.info(f"  area_m2 >= threshold cells: {(area_m2 >= min_area_m2).sum():,}")
    logger.info(f"  stream_mask cells (after dem_valid): {stream_mask.sum():,}")
    # Sample some off-network stream_mask cells
    off_network = stream_mask & ~dem_valid
    logger.info(f"  stream_mask cells outside dem_valid: {off_network.sum():,}")
    # Check FAC values at nodata DEM locations
    if nodata is not None:
        fac_at_nodata = fac[dem == nodata]
        logger.info(f"  FAC at DEM nodata cells — min: {fac_at_nodata.min():.0f}  max: {fac_at_nodata.max():.0f}  nonzero: {(fac_at_nodata != 0).sum():,}")

    # ------------------------------------------------------------------
    # 4. Load full-mosaic FDR, reprojected to this watershed's grid.
    #    Nearest-neighbour reproject of FDR is acceptable — we only need
    #    the pointer direction, not an exact accumulation value.
    # ------------------------------------------------------------------
    with rasterio.open(str(fdr_path)) as fdr_src:
        fdr_raw    = np.zeros(dem.shape, dtype=np.int32)
        fdr_nodata = fdr_src.nodata
        reproject(
            source=rasterio.band(fdr_src, 1),
            destination=fdr_raw,
            src_transform=fdr_src.transform,
            src_crs=fdr_src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
        )

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
    # 9. Build output GeoDataFrame
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
    wbt_dir    = Path(WBT_DIR)
    dems_dir   = Path(WATERSHED_DEMS_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    fdr_path = wbt_dir / FDR_FILE

    if not fdr_path.exists():
        logger.error(f"Flow direction not found: {fdr_path}")
        logger.error("Run wbt_hydrology.py first.")
        sys.exit(1)

    # Collect watershed DEMs, excluding the _fac.tif files
    dem_files = sorted(
        f for f in dems_dir.glob("watershed_*.tif")
        if not f.stem.endswith("_fac")
    )

    if not dem_files:
        logger.error(f"No watershed_*.tif files found in: {dems_dir}")
        logger.error("Run clip_watersheds.py first.")
        sys.exit(1)

    # Verify matching FAC files exist before starting
    missing_fac = [
        f for f in dem_files
        if not (dems_dir / f"{f.stem}_fac.tif").exists()
    ]
    if missing_fac:
        logger.error(
            f"{len(missing_fac)} watershed(s) missing a _fac.tif file: "
            f"{[f.stem for f in missing_fac]}. "
            f"Re-run clip_watersheds.py."
        )
        sys.exit(1)

    total = len(dem_files)
    logger.info(f"Found {total} watershed DEMs")
    logger.info(f"Input DEMs           : {dems_dir}")
    logger.info(f"Flow direction       : {fdr_path}")
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
