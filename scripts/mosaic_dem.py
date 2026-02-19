"""
mosaic_dem.py
-------------
Mosaics all 274 DEM tiles (gt_*.tif) into a single seamless DEM raster
for use as input to WhiteboxTools hydrology processing.

Uses the same batch mosaicking approach as mosaic_hydrology.py to avoid
Windows command line length limits.

USAGE:
    1. Edit the paths in the CONFIG section below.
    2. Run from the ArcGIS Pro Python environment:
       conda activate arcgispro-py3
       python mosaic_dem.py

Requirements:
    - ArcGIS Pro
"""

import logging
import sys
import time
from pathlib import Path

import arcpy

# Calculate the path to the project root (one level up from scripts/)
root_dir = Path(__file__).resolve().parent.parent

# Add the root directory to sys.path so Python can find config.py
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

# =============================================================================
# CONFIG — Edit these before running
# =============================================================================

DEM_DIR    = config.DATA_SCRATCH_DEMS             # Folder containing gt_*.tif DEM tiles
OUTPUT_DIR = config.DATA_DEM_MOSAIC               # Output folder
OUTPUT_FILE = "dem_mosaic.tif"                    # Output mosaic filename

# =============================================================================
# END CONFIG — No edits needed below this line
# =============================================================================


def setup_logging(output_dir: Path) -> logging.Logger:
    log_path = output_dir / "mosaic_dem.log"
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


def main():
    dem_dir    = Path(DEM_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    # Collect all DEM tiles
    dem_files = sorted(dem_dir.glob("gt_*.tif"))
    if not dem_files:
        logger.error(f"No gt_*.tif files found in: {dem_dir}")
        sys.exit(1)

    total    = len(dem_files)
    out_path = output_dir / OUTPUT_FILE

    logger.info(f"Found {total} DEM tiles")
    logger.info(f"Input dir  : {dem_dir}")
    logger.info(f"Output dir : {output_dir}")
    logger.info(f"Output file: {OUTPUT_FILE}")
    logger.info("-" * 60)

    if out_path.exists():
        logger.info(f"Output already exists — skipping: {out_path.name}")
        sys.exit(0)

    # Get spatial reference and cell size from first tile
    desc      = arcpy.Describe(str(dem_files[0]))
    cell_size = desc.meanCellWidth
    sr        = desc.spatialReference

    arcpy.env.overwriteOutput = True

    # Process in batches of 50 to avoid Windows command line length limits
    batch_size = 50
    batches    = [dem_files[i:i + batch_size]
                  for i in range(0, total, batch_size)]

    start_time = time.time()

    for b, batch in enumerate(batches, start=1):
        inputs = ";".join([str(p) for p in batch])
        logger.info(f"Batch {b}/{len(batches)} — {len(batch)} tiles...")

        if not out_path.exists():
            # First batch — create the output mosaic
            arcpy.management.MosaicToNewRaster(
                input_rasters      = inputs,
                output_location    = str(output_dir),
                raster_dataset_name_with_extension = OUTPUT_FILE,
                coordinate_system_for_the_raster   = sr,
                pixel_type         = "32_BIT_FLOAT",
                cellsize           = cell_size,
                number_of_bands    = 1,
                mosaic_method      = "LAST",
                mosaic_colormap_mode = "FIRST"
            )
        else:
            # Subsequent batches — mosaic into existing output
            arcpy.management.Mosaic(
                inputs      = inputs,
                target      = str(out_path),
                mosaic_type = "LAST",
                colormap    = "FIRST"
            )

        elapsed  = time.time() - start_time
        rate     = b / elapsed
        eta_min  = (len(batches) - b) / rate / 60 if rate > 0 else 0
        logger.info(f"Batch {b} complete | elapsed: {elapsed/60:.1f} min | ETA: {eta_min:.1f} min")

    elapsed_total = time.time() - start_time
    size_gb       = out_path.stat().st_size / 1024 ** 3

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info(f"  Output     : {out_path}")
    logger.info(f"  Size       : {size_gb:.2f} GB")
    logger.info(f"  Total time : {elapsed_total / 60:.1f} minutes")


if __name__ == "__main__":
    main()
