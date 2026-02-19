"""
calculate_ksn.py
----------------
Calculates normalized channel steepness index (ksn) for watershed DEMs
using direct raster operations with numpy/scipy/rasterio, without
dependency on topotoolbox.

Uses the WhiteboxTools flow accumulation and flow direction outputs
that are already computed for each watershed, extracting stream points
and calculating ksn = slope / (area^-theta) where theta = 0.45.

Exports results as point shapefiles with ksn values as attributes.

USAGE:
    1. Edit the paths and parameters in the CONFIG section below.
    2. Ensure these are already computed for each watershed:
       - DEM (from clip_watersheds.py)
       - Flow accumulation  
       - Flow direction
    3. Run from an environment with geopandas:
       conda activate demenv  # or any env with geopandas, rasterio, scipy
       python calculate_ksn.py

Requirements:
    conda install -c conda-forge geopandas rasterio scipy numpy shapely
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import generic_filter
import geopandas as gpd
from shapely.geometry import Point

# Calculate the path to the project root (one level up from scripts/)
root_dir = Path(__file__).resolve().parent.parent

# Add the root directory to sys.path so Python can find config.py
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

# =============================================================================
# CONFIG — Edit these before running
# =============================================================================

WBT_DIR            = config.DATA_SCRATCH_WBT                # WBT outputs (flow acc/dir)
WATERSHED_DEMS_DIR = config.DATA_SCRATCH_DEMS               # Watershed DEMs
OUTPUT_DIR         = config.DATA_KSN                        # Output ksn shapefiles

FAC_FILE = "flow_accumulation.tif"   # Flow accumulation from WBT
FDR_FILE = "flow_direction.tif"      # Flow direction from WBT (not actually needed)

# Ksn calculation parameters
MIN_DRAINAGE_AREA_M2 = config.MIN_DRAINAGE_AREA_M2
REFERENCE_CONCAVITY  = config.REFERENCE_CONCAVITY
SMOOTHING_WINDOW     = config.SMOOTHING_WINDOW
SAMPLE_DISTANCE      = config.SAMPLE_DISTANCE

# =============================================================================
# END CONFIG — No edits needed below this line
# =============================================================================


def setup_logging(output_dir: Path) -> logging.Logger:
    log_path = output_dir / "calculate_ksn.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def calculate_gradient_smoothed(dem: np.ndarray, cellsize: float,
                                 window_size: int = 5) -> np.ndarray:
    """
    Calculate slope using a smoothed gradient to reduce noise.
    Uses a moving window mean before computing gradient.
    Returns slope in m/m.
    """
    # Smooth DEM first using mean filter
    smoothed = generic_filter(dem, np.nanmean, size=window_size, mode='nearest')
    
    # Calculate gradients using central differences
    dy, dx = np.gradient(smoothed, cellsize)
    
    # Calculate slope magnitude
    slope = np.sqrt(dx**2 + dy**2)
    
    return slope


def extract_stream_points(dem_path: Path, fac_path: Path,
                          min_area_m2: float, sample_dist: float,
                          theta: float, window_size: int,
                          logger: logging.Logger) -> gpd.GeoDataFrame:
    """
    Extracts stream points with ksn values from a watershed DEM.
    Returns a GeoDataFrame with point geometry and attributes.
    """
    # Load DEM
    with rasterio.open(str(dem_path)) as src:
        dem      = src.read(1)
        transform = src.transform
        crs       = src.crs
        cellsize  = src.res[0]  # Assumes square cells
        nodata    = src.nodata
    
    # Mask nodata
    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)
    
    # Load flow accumulation
    with rasterio.open(str(fac_path)) as src:
        fac = src.read(1).astype(float)
    
    # Calculate drainage area in m²
    area_m2 = fac * (cellsize ** 2)
    
    # Extract stream mask (cells above threshold)
    stream_mask = area_m2 >= min_area_m2
    
    if not stream_mask.any():
        logger.warning(f"  No stream cells above threshold")
        return None
    
    # Calculate slope
    logger.info(f"  Computing slope...")
    slope = calculate_gradient_smoothed(dem, cellsize, window_size)
    
    # Calculate ksn = slope / (area^-theta)
    logger.info(f"  Computing ksn...")
    area_safe = np.maximum(area_m2, 1.0)  # Avoid division by zero
    ksn = slope / (area_safe ** (-theta))
    
    # Handle infinities and NaNs
    ksn = np.where(np.isfinite(ksn), ksn, 0)
    
    # Extract stream pixel coordinates and values
    rows, cols = np.where(stream_mask)
    
    # Convert row/col to coordinates
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    
    # Sample every N meters approximately
    # At 2m resolution, this means every sample_dist/2 pixels
    sample_step = max(1, int(sample_dist / cellsize))
    sampled_indices = np.arange(0, len(xs), sample_step)
    
    xs_sampled = [xs[i] for i in sampled_indices]
    ys_sampled = [ys[i] for i in sampled_indices]
    ksn_sampled = [ksn[rows[i], cols[i]] for i in sampled_indices]
    slope_sampled = [slope[rows[i], cols[i]] for i in sampled_indices]
    area_sampled = [area_m2[rows[i], cols[i]] / 1e6 for i in sampled_indices]  # Convert to km²
    
    # Build GeoDataFrame
    points = [Point(x, y) for x, y in zip(xs_sampled, ys_sampled)]
    gdf = gpd.GeoDataFrame({
        'ksn':      ksn_sampled,
        'slope':    slope_sampled,
        'area_km2': area_sampled
    }, geometry=points, crs=crs)
    
    return gdf


def calculate_ksn_for_watershed(dem_path: Path, fac_path: Path,
                                output_dir: Path, logger: logging.Logger) -> tuple[bool, str]:
    """
    Calculates ksn for a single watershed DEM and exports to shapefile.
    Returns (success, output_path).
    """
    watershed_id = dem_path.stem  # e.g. "watershed_1"
    out_shp      = output_dir / f"{watershed_id}_ksn.shp"
    
    # Skip if already exists
    if out_shp.exists():
        return (True, str(out_shp))
    
    try:
        logger.info(f"  Extracting stream points...")
        gdf = extract_stream_points(
            dem_path, fac_path,
            MIN_DRAINAGE_AREA_M2, SAMPLE_DISTANCE,
            REFERENCE_CONCAVITY, SMOOTHING_WINDOW,
            logger
        )
        
        if gdf is None or len(gdf) == 0:
            logger.warning(f"  No streams found — skipping")
            return (False, "")
        
        # Export to shapefile
        logger.info(f"  Exporting to shapefile...")
        gdf.to_file(str(out_shp))
        
        point_count = len(gdf)
        ksn_mean    = gdf['ksn'].mean()
        ksn_std     = gdf['ksn'].std()
        
        logger.info(
            f"  Exported {point_count} points  |  "
            f"ksn mean: {ksn_mean:.1f}  |  "
            f"ksn std: {ksn_std:.1f}"
        )
        
        return (True, str(out_shp))
        
    except Exception as e:
        logger.error(f"  Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        if out_shp.exists():
            try:
                for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                    f = out_shp.parent / (out_shp.stem + ext)
                    if f.exists():
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
    
    fac_path = wbt_dir / FAC_FILE
    
    # Validate inputs
    if not fac_path.exists():
        logger.error(f"Flow accumulation not found: {fac_path}")
        logger.error("Run wbt_hydrology.py first.")
        sys.exit(1)
    
    # Collect watershed DEMs
    dem_files = sorted(dems_dir.glob("watershed_*.tif"))
    if not dem_files:
        logger.error(f"No watershed_*.tif files found in: {dems_dir}")
        logger.error("Run clip_watersheds.py first.")
        sys.exit(1)
    
    total = len(dem_files)
    logger.info(f"Found {total} watershed DEMs")
    logger.info(f"Input DEMs           : {dems_dir}")
    logger.info(f"Flow accumulation    : {fac_path}")
    logger.info(f"Output dir           : {output_dir}")
    logger.info(f"Min drainage area    : {MIN_DRAINAGE_AREA_M2/1e6:.1f} km²")
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
        
        out_shp = output_dir / f"{watershed_id}_ksn.shp"
        if out_shp.exists():
            skipped += 1
            logger.info(f"[{i:3d}/{total}] SKIP  {watershed_id} — already exists")
            continue
        
        logger.info(f"[{i:3d}/{total}] START {watershed_id}")
        tile_start = time.time()
        
        success, result = calculate_ksn_for_watershed(
            dem_path, fac_path, output_dir, logger
        )
        
        if success:
            succeeded += 1
            tile_time = time.time() - tile_start
            elapsed   = time.time() - start_time
            rate      = i / elapsed
            eta_min   = (total - i) / rate / 60 if rate > 0 else 0
            
            logger.info(
                f"[{i:3d}/{total}] OK    {watershed_id}  |  "
                f"{tile_time:.1f}s  |  "
                f"ETA {eta_min:.1f} min"
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
    logger.info("Load *_ksn.shp files in ArcGIS Pro to visualize ksn values.")


if __name__ == "__main__":
    main()
