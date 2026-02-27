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
from shapely.geometry import shape

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

# Minimum number of pixels a stream polygon must contain to be written.
# Filters out single-pixel noise and tiny isolated patches.
# At 2m resolution, 5 pixels = 20 m² — adjust as needed.
MIN_PIXELS = 5

# Number of features to buffer before each fiona batch write.
# Larger values reduce I/O overhead on big networks.
BATCH_SIZE = 1000

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


def vectorize_streams(
    stream_mask: np.ndarray,
    transform,
    crs,
    out_path: Path,
    logger: logging.Logger,
    min_pixels: int = MIN_PIXELS,
    batch_size: int = BATCH_SIZE,
) -> int:
    """
    Vectorize a binary stream mask into polygon features and write to a
    GeoPackage.

    Key efficiency improvements over the original:
    - Uses a generator (not a list) for rasterio.features.shapes so shapes
      are processed one at a time without loading all into memory at once.
    - Filters tiny polygons by area before writing to cut output bloat.
    - Batches fiona writes (writerecords) instead of writing one feature at
      a time, significantly reducing I/O overhead on large networks.
    - Stores area_m2 as an attribute for convenient downstream filtering.

    Returns the number of features written.
    """
    mask_uint8 = stream_mask.astype(np.uint8)

    # Area of one raster pixel in map units² (handles non-square pixels)
    pixel_area = abs(transform.a * transform.e)
    min_area   = min_pixels * pixel_area

    schema = {
        "geometry": "Polygon",
        "properties": {
            "seg_id":  "int",
            "area_m2": "float",
        }
    }

    out_crs = crs.to_wkt() if crs else None

    # Remove existing output so fiona writes a fresh file
    if out_path.exists():
        out_path.unlink()

    count = 0
    batch = []

    # rasterio.features.shapes returns a generator — memory-efficient for
    # large rasters since shapes are yielded one at a time.
    shapes_gen = rasterio.features.shapes(
        mask_uint8,
        mask=mask_uint8,
        transform=transform,
        connectivity=8,
    )

    with fiona.open(
        str(out_path),
        mode="w",
        driver="GPKG",
        schema=schema,
        crs=out_crs,
        layer="streams",
    ) as dst:
        for geom_dict, value in shapes_gen:
            if value != 1:
                continue

            # Filter out tiny noise polygons before computing anything else
            geom = shape(geom_dict)
            if geom.area < min_area:
                continue

            count += 1
            batch.append({
                "geometry":   geom_dict,
                "properties": {"seg_id": count, "area_m2": round(geom.area, 2)},
            })

            # Flush batch to disk periodically to keep memory usage flat
            if len(batch) >= batch_size:
                dst.writerecords(batch)
                batch.clear()

        # Write any remaining features
        if batch:
            dst.writerecords(batch)

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
    logger.info(f"Min pixels: {MIN_PIXELS} (~{MIN_PIXELS * 4:.0f} m² at 2m)")
    logger.info("-" * 60)

    start_time = time.time()

    try:
        # --- Step 1: Read flow accumulation and apply threshold ---
        logger.info("Reading flow accumulation raster...")
        with rasterio.open(str(fac_path)) as fac_ds:
            # Use float32 instead of float64 — halves peak memory usage.
            # FAC values rarely need double precision; float32 handles up to
            # ~16.7 million exactly, and larger values with minor rounding.
            fac_data  = fac_ds.read(1).astype(np.float32)
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
            logger.error(
                "No stream pixels found at this threshold. "
                "Lower THRESHOLD or check input data."
            )
            sys.exit(1)

        # Free the full FAC array — no longer needed
        del fac_data, valid_mask

        # --- Step 2: Vectorize stream mask ---
        logger.info("Vectorizing stream mask...")
        logger.info("  (rasterio.features.shapes — typically seconds to minutes)")

        count = vectorize_streams(stream_mask, transform, crs, out_path, logger)
        logger.info(f"  Features written: {count:,}")

        if count == 0:
            logger.error(
                "No features were written. "
                "Check input data, threshold, and MIN_PIXELS setting."
            )
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
