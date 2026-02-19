"""
clip_watersheds.py
------------------
Clips the full DEM mosaic to each watershed polygon, producing individual
watershed DEMs suitable for ksn analysis in topotoolbox.

For each watershed polygon, extracts the corresponding DEM extent and saves
it as a separate GeoTIFF file named by the watershed ID.

USAGE:
    1. Edit the paths in the CONFIG section below.
    2. Run from the ArcGIS Pro Python environment:
       conda activate arcgispro-py3
       python clip_watersheds.py

Requirements:
    - ArcGIS Pro with Spatial Analyst extension
    - Completed delineate_watersheds.py first
"""

import logging
import sys
import time
from pathlib import Path

import arcpy
from arcpy.sa import ExtractByMask

# Calculate the path to the project root (one level up from scripts/)
root_dir = Path(__file__).resolve().parent.parent

# Add the root directory to sys.path so Python can find config.py
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

# =============================================================================
# CONFIG — Edit these before running
# =============================================================================

DEM_MOSAIC      = config.DATA_DEM_MOSAIC / "dem_mosaic.tif"     # Full DEM mosaic
WATERSHEDS_SHP  = config.DATA_SCRATCH_WATERSHEDS / "watersheds.shp"      # Watershed polygons
OUTPUT_DIR      = config.DATA_WATERSHEDS                 # Output folder for clipped DEMs

# Field in watersheds.shp that contains unique watershed IDs
# The script will use this to name output files
ID_FIELD = "gridcode"    # Default field created by RasterToPolygon

# =============================================================================
# END CONFIG — No edits needed below this line
# =============================================================================


def setup_logging(output_dir: Path) -> logging.Logger:
    log_path = output_dir / "clip_watersheds.log"
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


def clip_watershed(dem_path: str, watershed_geom, watershed_id: int,
                   output_dir: Path, logger: logging.Logger) -> tuple[bool, str]:
    """
    Clips the DEM to a single watershed polygon.
    Returns (success, output_path).
    """
    out_path = output_dir / f"watershed_{watershed_id}.tif"

    # Skip if already exists
    if out_path.exists():
        return (True, str(out_path))

    try:
        # Use ExtractByMask to clip DEM to watershed polygon
        clipped = ExtractByMask(dem_path, watershed_geom)
        clipped.save(str(out_path))

        size_mb = out_path.stat().st_size / 1024 / 1024
        return (True, str(out_path))

    except Exception as e:
        logger.error(f"  Failed to clip watershed {watershed_id}: {e}")
        if out_path.exists():
            try:
                arcpy.management.Delete(str(out_path))
            except Exception:
                pass
        return (False, "")


def main():
    dem_path       = Path(DEM_MOSAIC)
    watersheds_shp = Path(WATERSHEDS_SHP)
    output_dir     = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    # Validate inputs
    if not dem_path.exists():
        logger.error(f"DEM mosaic not found: {dem_path}")
        sys.exit(1)
    if not watersheds_shp.exists():
        logger.error(f"Watersheds shapefile not found: {watersheds_shp}")
        logger.error("Run delineate_watersheds.py first.")
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

    # Count watersheds
    total = int(arcpy.management.GetCount(str(watersheds_shp))[0])
    logger.info(f"Found {total} watersheds")
    logger.info(f"DEM mosaic : {dem_path}")
    logger.info(f"Output dir : {output_dir}")
    logger.info("-" * 60)

    start_time = time.time()
    succeeded  = 0
    failed     = 0
    skipped    = 0

    # Process each watershed
    with arcpy.da.SearchCursor(str(watersheds_shp), ["SHAPE@", ID_FIELD]) as cursor:
        for i, (geom, wid) in enumerate(cursor, start=1):
            out_path = output_dir / f"watershed_{wid}.tif"

            if out_path.exists():
                skipped += 1
                logger.info(f"[{i:3d}/{total}] SKIP  Watershed {wid} — already exists")
                continue

            logger.info(f"[{i:3d}/{total}] START Watershed {wid}...")
            tile_start = time.time()

            success, result_path = clip_watershed(
                str(dem_path), geom, wid, output_dir, logger
            )

            if success:
                succeeded += 1
                tile_time = time.time() - tile_start
                size_mb   = Path(result_path).stat().st_size / 1024 / 1024

                elapsed = time.time() - start_time
                rate    = i / elapsed
                eta_min = (total - i) / rate / 60 if rate > 0 else 0

                logger.info(
                    f"[{i:3d}/{total}] OK    Watershed {wid}  |  "
                    f"{size_mb:.1f} MB  |  "
                    f"{tile_time:.1f}s  |  "
                    f"ETA {eta_min:.1f} min"
                )
            else:
                failed += 1
                logger.error(f"[{i:3d}/{total}] FAIL  Watershed {wid}")

    arcpy.CheckInExtension("Spatial")

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
    logger.info("Watershed DEMs ready for ksn analysis in topotoolbox.")


if __name__ == "__main__":
    main()
