"""
stream_extraction_wbt.py
------------------------
Extracts a fully connected stream network from WhiteboxTools flow
accumulation output. Uses arcpy's Stream to Feature tool to convert
the thresholded stream raster to vector polylines.

USAGE:
    1. Edit the paths and threshold in the CONFIG section below.
    2. Run from the ArcGIS Pro Python environment:
       conda activate arcgispro-py3
       python stream_extraction_wbt.py

Requirements:
    - ArcGIS Pro with Spatial Analyst extension
    - Completed wbt_hydrology.py first
"""

import logging
import sys
import time
from pathlib import Path

import arcpy
from arcpy.sa import Con, Raster, Int

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
FDR_FILE    = "flow_direction.tif"      # Flow direction from WBT
OUTPUT_FILE = "streams_connected.shp"   # Output stream network

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


def main():
    wbt_dir    = Path(WBT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    fac_path = wbt_dir / FAC_FILE
    fdr_path = wbt_dir / FDR_FILE
    out_shp  = output_dir / OUTPUT_FILE

    # Validate inputs
    if not fac_path.exists():
        logger.error(f"Flow accumulation not found: {fac_path}")
        logger.error("Run wbt_hydrology.py first.")
        sys.exit(1)
    if not fdr_path.exists():
        logger.error(f"Flow direction not found: {fdr_path}")
        logger.error("Run wbt_hydrology.py first.")
        sys.exit(1)

    # Check Spatial Analyst license
    if arcpy.CheckExtension("Spatial") == "Available":
        arcpy.CheckOutExtension("Spatial")
        logger.info("Spatial Analyst extension checked out.")
    else:
        logger.error("Spatial Analyst extension not available. Exiting.")
        sys.exit(1)

    arcpy.env.overwriteOutput = True
    arcpy.env.workspace       = str(output_dir)

    logger.info(f"Threshold: {THRESHOLD:,} cells (~{THRESHOLD * 4 / 1e6:.1f} km² at 2m)")
    logger.info("-" * 60)

    start_time = time.time()

    try:
        # --- Step 1: Apply threshold to create binary stream raster ---
        logger.info("Applying threshold to flow accumulation...")
        fac_raster    = Raster(str(fac_path))
        stream_raster = Con(fac_raster >= THRESHOLD, 1)

        temp_stream = str(output_dir / "tmp_stream_raster.tif")
        stream_raster.save(temp_stream)
        logger.info("Stream raster created.")

        # --- Step 2: Convert WBT flow direction to integer (required by StreamToFeature) ---
        # WhiteboxTools D8Pointer outputs float, but arcpy requires integer
        logger.info("Converting flow direction to integer...")
        fdr_int  = Int(Raster(str(fdr_path)))
        temp_fdr = str(output_dir / "tmp_fdr_int.tif")
        fdr_int.save(temp_fdr)
        logger.info("Flow direction integer raster created.")

        # --- Step 3: Convert to vector polylines ---
        logger.info("Converting stream raster to vector polylines...")
        logger.info("(This may take several minutes for the full study area)")

        arcpy.sa.StreamToFeature(
            in_stream_raster         = temp_stream,
            in_flow_direction_raster = temp_fdr,
            out_polyline_features    = str(out_shp),
            simplify                 = "SIMPLIFY"
        )

        # Clean up temp rasters
        arcpy.management.Delete(temp_stream)
        arcpy.management.Delete(temp_fdr)
        logger.info("Temporary rasters deleted.")

        if not out_shp.exists():
            raise RuntimeError("Output shapefile was not created.")

        # Report statistics
        result = arcpy.management.GetCount(str(out_shp))
        count  = int(result.getOutput(0))

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("COMPLETE")
        logger.info(f"  Output        : {out_shp}")
        logger.info(f"  Stream count  : {count:,} segments")
        logger.info(f"  Total time    : {elapsed / 60:.1f} minutes")
        logger.info("")
        logger.info("Load streams_connected.shp in ArcGIS Pro to verify the network.")

    except Exception as e:
        logger.error(f"FAILED: {e}")
        arcpy.CheckInExtension("Spatial")
        sys.exit(1)

    arcpy.CheckInExtension("Spatial")


if __name__ == "__main__":
    main()
