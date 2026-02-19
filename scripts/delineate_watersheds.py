"""
delineate_watersheds.py
-----------------------
Automatically identifies stream outlet points where streams exit the study
area or reach a drainage area threshold, then delineates the contributing
watershed for each outlet using the WhiteboxTools flow direction raster.

This produces a set of watershed polygons suitable for clipping DEMs and
running ksn analysis on individual drainages.

USAGE:
    1. Edit the paths and threshold in the CONFIG section below.
    2. Run from the ArcGIS Pro Python environment:
       conda activate arcgispro-py3
       python delineate_watersheds.py

Requirements:
    - ArcGIS Pro with Spatial Analyst extension
    - Completed wbt_hydrology.py and stream_extraction_wbt.py first
"""

import logging
import sys
import time
from pathlib import Path

import arcpy
from arcpy.sa import Raster, SnapPourPoint, Watershed

# Calculate the path to the project root (one level up from scripts/)
root_dir = Path(__file__).resolve().parent.parent

# Add the root directory to sys.path so Python can find config.py
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

# =============================================================================
# CONFIG — Edit these before running
# =============================================================================

WBT_DIR         = config.DATA_SCRATCH_WBT                    # Folder with WBT outputs
STREAMS_SHP     = config.DATA_STREAMS                        # Stream network
OUTPUT_DIR      = config.DATA_SCRATCH_WATERSHEDS             # Output folder

FAC_FILE        = "flow_accumulation.tif"   # Flow accumulation from WBT
FDR_FILE        = "flow_direction.tif"      # Flow direction from WBT

# Minimum drainage area threshold for watershed outlets
# Only stream segments with drainage area >= this value will get watersheds
# At 2m resolution:
#   10,000,000 cells  = ~40 km²   (large watersheds only)
#   25,000,000 cells  = ~100 km²  (major drainages)
#   50,000,000 cells  = ~200 km²  (very large basins)
MIN_DRAINAGE_AREA_CELLS = config.MIN_DRAINAGE_AREA_CELLS    # ~40 km² at 2m resolution

# Snap distance for pour points (cells)
# Pour points are snapped to the highest flow accumulation cell within
# this distance to ensure they land exactly on the stream
SNAP_DISTANCE = config.SNAP_DISTANCE    # cells (100m at 2m resolution)

# =============================================================================
# END CONFIG — No edits needed below this line
# =============================================================================


def setup_logging(output_dir: Path) -> logging.Logger:
    log_path = output_dir / "delineate_watersheds.log"
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
    wbt_dir     = Path(WBT_DIR)
    output_dir  = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    fac_path    = wbt_dir / FAC_FILE
    fdr_path    = wbt_dir / FDR_FILE
    streams_shp = Path(STREAMS_SHP)

    # Validate inputs
    if not fac_path.exists():
        logger.error(f"Flow accumulation not found: {fac_path}")
        sys.exit(1)
    if not fdr_path.exists():
        logger.error(f"Flow direction not found: {fdr_path}")
        sys.exit(1)
    if not streams_shp.exists():
        logger.error(f"Stream network not found: {streams_shp}")
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

    logger.info(f"Min drainage area: {MIN_DRAINAGE_AREA_CELLS:,} cells "
                f"(~{MIN_DRAINAGE_AREA_CELLS * 4 / 1e6:.1f} km²)")
    logger.info(f"Snap distance: {SNAP_DISTANCE} cells")
    logger.info("-" * 60)

    start_time = time.time()

    try:
        # --- Step 1: Identify stream outlet points ---
        # Stream outlets are endpoints of stream polylines where the
        # drainage area exceeds our threshold
        logger.info("Step 1: Identifying stream outlet points...")

        # Convert stream endpoints to points
        endpoints_fc = str(output_dir / "stream_endpoints.shp")
        arcpy.management.FeatureVerticesToPoints(
            in_features  = str(streams_shp),
            out_feature_class = endpoints_fc,
            point_location = "END"
        )
        logger.info(f"  Extracted {arcpy.management.GetCount(endpoints_fc)[0]} stream endpoints")

        # Extract flow accumulation values at each endpoint
        logger.info("  Extracting flow accumulation at endpoints...")
        arcpy.sa.ExtractValuesToPoints(
            in_point_features = endpoints_fc,
            in_raster         = str(fac_path),
            out_point_features = str(output_dir / "endpoints_with_fac.shp"),
            interpolate_values = "NONE"
        )

        # Select only endpoints above threshold
        outlets_fc = str(output_dir / "outlets.shp")
        where_clause = f"RASTERVALU >= {MIN_DRAINAGE_AREA_CELLS}"
        arcpy.analysis.Select(
            in_features  = str(output_dir / "endpoints_with_fac.shp"),
            out_feature_class = outlets_fc,
            where_clause = where_clause
        )

        outlet_count = int(arcpy.management.GetCount(outlets_fc)[0])
        logger.info(f"  Found {outlet_count} outlets above threshold")

        if outlet_count == 0:
            logger.error("No outlets found above threshold. Lower MIN_DRAINAGE_AREA_CELLS.")
            sys.exit(1)

        # --- Step 2: Snap pour points to highest flow accumulation ---
        logger.info("Step 2: Snapping pour points to stream cells...")
        snapped_fc = str(output_dir / "pourpoints_snapped.shp")
        fac_raster = Raster(str(fac_path))

        snapped = SnapPourPoint(
            in_pour_point_data = outlets_fc,
            in_accumulation_raster = fac_raster,
            snap_distance = SNAP_DISTANCE
        )
        snapped.save(snapped_fc)
        logger.info(f"  Pour points snapped")

        # --- Step 3: Delineate watersheds ---
        logger.info("Step 3: Delineating watersheds...")
        fdr_raster = Raster(str(fdr_path))

        watersheds_raster = Watershed(
            in_flow_direction_raster = fdr_raster,
            in_pour_point_data       = snapped_fc
        )

        watersheds_tif = str(output_dir / "watersheds.tif")
        watersheds_raster.save(watersheds_tif)
        logger.info("  Watershed raster created")

        # --- Step 4: Convert to polygons ---
        logger.info("Step 4: Converting watersheds to polygons...")
        watersheds_poly = str(output_dir / "watersheds.shp")

        arcpy.conversion.RasterToPolygon(
            in_raster      = watersheds_tif,
            out_polygon_features = watersheds_poly,
            simplify_polygons    = "SIMPLIFY"
        )

        watershed_count = int(arcpy.management.GetCount(watersheds_poly)[0])
        logger.info(f"  Created {watershed_count} watershed polygons")

        # --- Step 5: Calculate area statistics ---
        logger.info("Step 5: Calculating watershed areas...")
        arcpy.management.AddGeometryAttributes(
            Input_Features = watersheds_poly,
            Geometry_Properties = "AREA",
            Area_Unit = "SQUARE_KILOMETERS"
        )

        # Clean up intermediate files
        logger.info("Cleaning up intermediate files...")
        for temp_file in [endpoints_fc, str(output_dir / "endpoints_with_fac.shp"),
                          outlets_fc, snapped_fc, watersheds_tif]:
            try:
                arcpy.management.Delete(temp_file)
            except Exception:
                pass

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("COMPLETE")
        logger.info(f"  Output          : {watersheds_poly}")
        logger.info(f"  Watershed count : {watershed_count}")
        logger.info(f"  Total time      : {elapsed / 60:.1f} minutes")
        logger.info("")
        logger.info("Load watersheds.shp in ArcGIS Pro to visualize.")
        logger.info("Use these polygons to clip DEMs for individual ksn analysis.")

    except Exception as e:
        logger.error(f"FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        arcpy.CheckInExtension("Spatial")
        sys.exit(1)

    arcpy.CheckInExtension("Spatial")


if __name__ == "__main__":
    main()
