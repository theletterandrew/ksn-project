"""
mosaic_dem.py
-------------
Mosaics all 274 DEM tiles (gt_*.tif) into a single seamless DEM raster
for use as input to WhiteboxTools hydrology processing.

Uses GDAL (BuildVRT + Translate) instead of ArcGIS Pro, eliminating the
ArcGIS Pro / arcpy dependency entirely. GDAL's VRT approach also avoids
Windows command-line length limits without needing to batch tiles manually.

USAGE:
    1. Edit the paths in the CONFIG section below.
    2. Run from any Python environment with GDAL installed:
       conda activate <your-env>
       python mosaic_dem.py

Requirements:
    - GDAL (osgeo) — e.g. via: conda install -c conda-forge gdal
    - config.py in the project root (same as before)
"""

import logging
import sys
import time
from pathlib import Path

from osgeo import gdal

gdal.UseExceptions()  # Raise Python exceptions on GDAL errors

# Calculate the path to the project root (one level up from scripts/)
root_dir = Path(__file__).resolve().parent.parent

# Add the root directory to sys.path so Python can find config.py
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

# =============================================================================
# CONFIG — Edit these before running
# =============================================================================

DEM_DIR     = config.DATA_SCRATCH_DEMS   # Folder containing gt_*.tif DEM tiles
OUTPUT_DIR  = config.DATA_DEM_MOSAIC     # Output folder
OUTPUT_FILE = "dem_mosaic.tif"           # Output mosaic filename

# =============================================================================
# END CONFIG — No edits needed below this line
# =============================================================================

NODATA_VALUE = -9999.0


def setup_logging(output_dir: Path) -> logging.Logger:
    log_path = output_dir / "mosaic_dem.log"
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

    total     = len(dem_files)
    out_path  = output_dir / OUTPUT_FILE
    vrt_path  = output_dir / "dem_mosaic.vrt"
    clean_path = output_dir / "dem_mosaic_wbt.tif"

    logger.info(f"Found {total} DEM tiles")
    logger.info(f"Input dir  : {dem_dir}")
    logger.info(f"Output dir : {output_dir}")
    logger.info(f"Output file: {OUTPUT_FILE}")
    logger.info("-" * 60)

    if out_path.exists():
        logger.info(f"Output already exists — skipping mosaic: {out_path.name}")
    else:
        start_time = time.time()

        # ------------------------------------------------------------------
        # Step 1: Build a VRT (virtual mosaic) from all tiles.
        # BuildVRT accepts a Python list, so no command-line length issues.
        # ------------------------------------------------------------------
        logger.info("Building VRT index...")
        vrt_options = gdal.BuildVRTOptions(
            resolution="highest",
            resampleAlg="nearest",
            outputSRS=None,   # Inherit SRS from input tiles
            VRTNodata=NODATA_VALUE,
        )
        vrt_ds = gdal.BuildVRT(
            str(vrt_path),
            [str(p) for p in dem_files],
            options=vrt_options,
        )
        if vrt_ds is None:
            logger.error("BuildVRT failed — check that input tiles are valid GeoTIFFs.")
            sys.exit(1)
        vrt_ds.FlushCache()
        vrt_ds = None  # Close dataset
        logger.info(f"VRT written: {vrt_path.name}")

        # ------------------------------------------------------------------
        # Step 2: Translate VRT → single GeoTIFF mosaic.
        # Using LZW compression and tiling for efficient storage/access.
        # ------------------------------------------------------------------
        logger.info("Translating VRT -> GeoTIFF mosaic (this may take a while)...")
        translate_options = gdal.TranslateOptions(
            format="GTiff",
            outputType=gdal.GDT_Float32,
            noData=NODATA_VALUE,
            creationOptions=[
                "COMPRESS=LZW",
                "TILED=YES",
                "BIGTIFF=IF_SAFER",
            ],
        )
        ds = gdal.Translate(
            str(out_path),
            str(vrt_path),
            options=translate_options,
        )
        if ds is None:
            logger.error("Translate failed.")
            sys.exit(1)
        ds.FlushCache()
        ds = None

        elapsed = time.time() - start_time
        size_gb = out_path.stat().st_size / 1024 ** 3
        logger.info(f"Mosaic complete | elapsed: {elapsed / 60:.1f} min | size: {size_gb:.2f} GB")

    # --------------------------------------------------------------------------
    # Step 3: Export a clean, uncompressed Float32 GeoTIFF for WhiteboxTools.
    # WhiteboxTools works best with simple, uncompressed GeoTIFFs.
    # --------------------------------------------------------------------------
    if clean_path.exists():
        logger.info(f"Clean WBT output already exists — skipping: {clean_path.name}")
    else:
        logger.info("Exporting clean GeoTIFF for WhiteboxTools...")
        wbt_options = gdal.TranslateOptions(
            format="GTiff",
            outputType=gdal.GDT_Float32,
            noData=NODATA_VALUE,
            creationOptions=[
                "COMPRESS=NONE",
                "BIGTIFF=IF_SAFER",
            ],
        )
        ds = gdal.Translate(
            str(clean_path),
            str(out_path),
            options=wbt_options,
        )
        if ds is None:
            logger.error("Clean GeoTIFF export failed.")
            sys.exit(1)
        ds.FlushCache()
        ds = None
        logger.info(f"Clean GeoTIFF written: {clean_path.name}")

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info(f"  Mosaic     : {out_path}")
    logger.info(f"  WBT input  : {clean_path}")


if __name__ == "__main__":
    main()
