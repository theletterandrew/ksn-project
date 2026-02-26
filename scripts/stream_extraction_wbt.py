"""
stream_extraction_wbt.py
------------------------
Extracts a fully connected stream network from WhiteboxTools flow
accumulation output. Converts the thresholded stream raster to vector
features using open-source Python GIS libraries (no arcpy required).

USAGE:
    1. Install dependencies:
       pip install rasterio numpy fiona shapely

    2. Edit the paths and threshold in the CONFIG section below.

    3. Run:
       python stream_extraction_wbt.py

Requirements:
    - rasterio
    - numpy
    - fiona
    - shapely
    - Completed wbt_hydrology.py first
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
import fiona
import fiona.crs

# Calculate the path to the project root (one level up from scripts/)
root_dir = Path(__file__).resolve().parent.parent

# Add the root directory to sys.path so Python can find config.py
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

# =============================================================================
# CONFIG — Edit these before running
# =============================================================================

WBT_DIR     = config.DATA_SCRATCH_WBT   # Folder with WBT outputs
OUTPUT_DIR  = config.DATA_STREAMS       # Output folder for streams

FAC_FILE    = "flow_accumulation.tif"   # Flow accumulation from WBT
OUTPUT_FILE = "streams_connected.gpkg"  # Output stream network (GeoPackage)

# Drainage area threshold
# Since this is from the full continuous mosaic, flow accumulates across
# the entire study area without tile boundary resets. Higher thresholds
# are now appropriate to avoid overly dense networks.
# At 2m resolution:
#   500,000 cells   = ~2 km²   (dense network)
#   1,000,000 cells = ~4 km²   (moderate)
#   2,500,000 cells = ~10 km²  (major channels only)
THRESHOLD = config.STREAM_THRESHOLD  # cells (~4 km² at 2m resolution)

# =============================================================================
# END CONFIG — No edits needed below this line
# =============================================================================


def setup_logging(output_dir: Path) -> logging.Logger:
    log_path = output_dir / "stream_extraction_wbt.log"
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


def vectorize_streams(stream_mask: np.ndarray, transform, crs, out_path: Path,
                      logger: logging.Logger) -> int:
    """
    Vectorize a binary stream mask into polygon features using
    rasterio.features.shapes, then write to a GeoPackage.

    rasterio.features.shapes runs in C and handles large rasters efficiently —
    far faster than pixel-by-pixel Python tracing. It produces polygon rings
    around connected pixel groups, giving the stream corridor footprint at
    raster resolution.

    Returns the number of features written.
    """
    # rasterio.features.shapes requires uint8 input
    mask_uint8 = stream_mask.astype(np.uint8)

    # Extract shapes only where mask == 1, using 8-connectivity
    shapes = list(rasterio.features.shapes(
        mask_uint8,
        mask=mask_uint8,
        transform=transform,
        connectivity=8
    ))

    if not shapes:
        return 0

    schema = {
        "geometry": "Polygon",
        "properties": {"seg_id": "int"}
    }

    out_crs = crs.to_wkt() if crs else None

    # Remove existing file so fiona writes fresh
    if out_path.exists():
        out_path.unlink()

    count = 0
    with fiona.open(
        str(out_path),
        mode="w",
        driver="GPKG",
        schema=schema,
        crs=out_crs,
        layer="streams"
    ) as dst:
        for geom_dict, value in shapes:
            if value != 1:
                continue
            count += 1
            dst.write({
                "geometry": geom_dict,
                "properties": {"seg_id": count}
            })

    return count


def main():
    wbt_dir    = Path(WBT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    fac_path = wbt_dir / FAC_FILE
    out_path = output_dir / OUTPUT_FILE

    # Validate inputs
    if not fac_path.exists():
        logger.error(f"Flow accumulation not found: {fac_path}")
        logger.error("Run wbt_hydrology.py first.")
        sys.exit(1)

    logger.info(f"Threshold : {THRESHOLD:,} cells (~{THRESHOLD * 4 / 1e6:.1f} km² at 2m)")
    logger.info("-" * 60)

    start_time = time.time()

    try:
        # --- Step 1: Read flow accumulation and apply threshold ---
        logger.info("Reading flow accumulation raster...")
        with rasterio.open(str(fac_path)) as fac_ds:
            fac_data  = fac_ds.read(1).astype(np.float64)
            transform = fac_ds.transform
            crs       = fac_ds.crs
            nodata    = fac_ds.nodata

        if nodata is not None:
            valid_mask = fac_data != nodata
        else:
            valid_mask = np.ones_like(fac_data, dtype=bool)

        logger.info("Applying threshold to flow accumulation...")
        stream_mask = (fac_data >= THRESHOLD) & valid_mask
        pixel_count = int(stream_mask.sum())
        logger.info(f"  Stream pixels above threshold: {pixel_count:,}")

        if pixel_count == 0:
            logger.error("No stream pixels found at this threshold. "
                         "Lower THRESHOLD or check input data.")
            sys.exit(1)

        # Free the full FAC array now that we have the mask
        del fac_data

        # --- Step 2: Vectorize stream mask ---
        logger.info("Vectorizing stream mask...")
        logger.info("  (rasterio.features.shapes — typically seconds to minutes)")

        count = vectorize_streams(stream_mask, transform, crs, out_path, logger)
        logger.info(f"  Features written: {count:,}")

        if count == 0:
            logger.error("No features were written. Check input data and threshold.")
            sys.exit(1)

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("COMPLETE")
        logger.info(f"  Output      : {out_path}")
        logger.info(f"  Features    : {count:,}")
        logger.info(f"  Total time  : {elapsed / 60:.1f} minutes")
        logger.info("")
        logger.info("Load streams_connected.gpkg in QGIS or ArcGIS Pro to verify.")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
