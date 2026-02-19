"""
las_to_dem.py
-------------
Batch rasterizes LAS/LAZ tiles to GeoTIFF DEMs using laspy, scipy, and rasterio.
Reprojects point coordinates from EPSG:3857 (WGS84 Pseudo-Mercator) to
EPSG:26911 (NAD83 / UTM Zone 11N) before gridding, so output DEMs are
in a metric projection suitable for geomorphic analysis.

Designed for the USGS_LPC_CA_SoCal_Wildfires_B1_2018 dataset.

USAGE:
    1. Edit the paths in the CONFIG section below.
    2. Run: python las_to_dem.py

Requirements:
    conda install -c conda-forge laspy lazrs-python scipy rasterio pyproj
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import laspy
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pyproj import Transformer
from scipy.spatial import cKDTree

# Calculate the path to the project root (one level up from scripts/)
root_dir = Path(__file__).resolve().parent.parent

# Add the root directory to sys.path so Python can find config.py
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

# =============================================================================
# CONFIG — Edit these before running
# =============================================================================

INPUT_DIR     = config.DATA_PROCESSED      # Folder containing .las or .laz files
OUTPUT_DIR    = config.DATA_SCRATCH_DEMS   # Folder for output GeoTIFFs

INPUT_CRS     = "EPSG:3857"   # Source CRS of the LAS/LAZ files
OUTPUT_CRS    = "EPSG:26911"  # Target CRS for output GeoTIFFs
                              # EPSG:26911 = NAD83 / UTM Zone 11N (metric, SoCal standard)

RESOLUTION    = 2.0       # Output raster resolution in meters
NODATA_VALUE  = -9999.0   # NoData fill value
IDW_POWER     = 2         # IDW distance weighting power for gap filling
IDW_NEIGHBORS = 8         # Number of nearest neighbors for IDW gap filling

# =============================================================================
# END CONFIG — No edits needed below this line
# =============================================================================


def setup_logging(output_dir: Path) -> logging.Logger:
    log_path = output_dir / "las_to_dem.log"
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


def las_to_dem(las_path: Path, out_path: Path, logger: logging.Logger) -> None:
    """
    Reads a LAS/LAZ file, reprojects XY coordinates from INPUT_CRS to
    OUTPUT_CRS, grids to a DEM using mean gridding with IDW gap fill,
    and writes a GeoTIFF.
    """

    # --- Read LAS/LAZ file ---
    logger.info(f"  Reading {las_path.name}...")
    las = laspy.read(str(las_path))

    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)

    if len(x) == 0:
        raise RuntimeError("No points found in file")

    logger.info(f"  Points: {len(x):,}  |  Z range: {z.min():.2f} to {z.max():.2f} m")

    # --- Reproject XY from INPUT_CRS to OUTPUT_CRS ---
    logger.info(f"  Reprojecting {INPUT_CRS} -> {OUTPUT_CRS}...")
    transformer = Transformer.from_crs(INPUT_CRS, OUTPUT_CRS, always_xy=True)
    x, y = transformer.transform(x, y)

    logger.info(f"  Reprojected X range: {x.min():.2f} to {x.max():.2f}")
    logger.info(f"  Reprojected Y range: {y.min():.2f} to {y.max():.2f}")

    # --- Define output grid ---
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    ncols = int(np.ceil((x_max - x_min) / RESOLUTION)) + 1
    nrows = int(np.ceil((y_max - y_min) / RESOLUTION)) + 1

    col_idx = np.floor((x - x_min) / RESOLUTION).astype(int)
    row_idx = np.floor((y_max - y) / RESOLUTION).astype(int)  # flip Y: top-left origin

    col_idx = np.clip(col_idx, 0, ncols - 1)
    row_idx = np.clip(row_idx, 0, nrows - 1)

    # --- Mean gridding ---
    logger.info(f"  Gridding to {nrows} x {ncols} raster...")
    z_sum   = np.zeros((nrows, ncols), dtype=np.float64)
    z_count = np.zeros((nrows, ncols), dtype=np.int32)

    np.add.at(z_sum,   (row_idx, col_idx), z)
    np.add.at(z_count, (row_idx, col_idx), 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(z_count > 0, z_sum / z_count, np.nan)

    # --- IDW gap fill ---
    empty_mask = np.isnan(grid)
    n_empty = empty_mask.sum()

    if n_empty > 0:
        logger.info(f"  Gap filling {n_empty:,} empty cells (IDW k={IDW_NEIGHBORS})...")

        filled_rows, filled_cols = np.where(~empty_mask)
        filled_z = grid[filled_rows, filled_cols]
        empty_rows, empty_cols = np.where(empty_mask)

        filled_xy = np.column_stack([filled_cols, filled_rows])
        empty_xy  = np.column_stack([empty_cols,  empty_rows])

        tree = cKDTree(filled_xy)
        distances, indices = tree.query(empty_xy, k=IDW_NEIGHBORS, workers=-1)

        distances = np.where(distances == 0, 1e-10, distances)
        weights   = 1.0 / (distances ** IDW_POWER)
        weights  /= weights.sum(axis=1, keepdims=True)

        grid[empty_rows, empty_cols] = (weights * filled_z[indices]).sum(axis=1)

    # Replace any remaining NaN with NoData
    grid = np.where(np.isnan(grid), NODATA_VALUE, grid).astype(np.float32)

    # --- Write GeoTIFF ---
    crs       = CRS.from_epsg(int(OUTPUT_CRS.split(":")[1]))
    transform = from_bounds(x_min, y_min, x_max, y_max, ncols, nrows)

    with rasterio.open(
        str(out_path),
        "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=NODATA_VALUE,
        compress="deflate"
    ) as dst:
        dst.write(grid, 1)

    size_mb = out_path.stat().st_size / 1024 / 1024
    logger.info(f"  Written: {out_path.name}  ({size_mb:.1f} MB)")


def main():
    input_dir  = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    # Accept both .las and .laz files
    las_files = sorted(list(input_dir.glob("*.las")) + list(input_dir.glob("*.laz")))
    if not las_files:
        logger.error(f"No .las or .laz files found in: {input_dir}")
        sys.exit(1)

    total = len(las_files)
    logger.info(f"Found {total} files")
    logger.info(f"Input dir  : {input_dir}")
    logger.info(f"Output dir : {output_dir}")
    logger.info(f"Input CRS  : {INPUT_CRS}")
    logger.info(f"Output CRS : {OUTPUT_CRS}")
    logger.info(f"Resolution : {RESOLUTION} m")
    logger.info("-" * 60)

    succeeded, failed, skipped = 0, 0, 0
    failures   = []
    start_time = time.time()

    for i, las_path in enumerate(las_files, start=1):
        out_path = output_dir / (las_path.stem + ".tif")

        if out_path.exists():
            skipped += 1
            logger.info(f"[{i:4d}/{total}] SKIP  {las_path.name} — already exists")
            continue

        logger.info(f"[{i:4d}/{total}] START {las_path.name}")
        tile_start = time.time()

        try:
            las_to_dem(las_path, out_path, logger)

            succeeded += 1
            tile_time = time.time() - tile_start
            elapsed   = time.time() - start_time
            eta_min   = (total - i) / (i / elapsed) / 60 if elapsed > 0 else 0

            logger.info(
                f"[{i:4d}/{total}] OK    {las_path.name}  |  "
                f"tile: {tile_time:.1f}s  |  "
                f"elapsed: {elapsed/60:.1f} min  |  "
                f"ETA: {eta_min:.1f} min"
            )

        except Exception as e:
            failed += 1
            failures.append((las_path.name, str(e)))
            if out_path.exists():
                out_path.unlink()
            logger.error(f"[{i:4d}/{total}] FAIL  {las_path.name} — {e}")

    # Final summary
    elapsed_total = time.time() - start_time
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info(f"  Succeeded : {succeeded}")
    logger.info(f"  Skipped   : {skipped}")
    logger.info(f"  Failed    : {failed}")
    logger.info(f"  Total time: {elapsed_total / 60:.1f} minutes")

    if failures:
        logger.error("Failed tiles:")
        for fname, msg in failures:
            logger.error(f"  {fname}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
