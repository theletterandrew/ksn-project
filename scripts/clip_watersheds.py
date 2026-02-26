"""
clip_watersheds.py
------------------
Clips the full DEM mosaic to each watershed polygon, producing individual
watershed DEMs suitable for ksn analysis in topotoolbox.

For each watershed polygon, extracts the corresponding DEM extent and saves
it as a separate GeoTIFF file named by the watershed ID.

USAGE:
    1. Install dependencies:
       pip install rasterio fiona shapely numpy

    2. Edit the paths in the CONFIG section below.

    3. Run:
       python clip_watersheds.py

Requirements:
    - rasterio
    - fiona
    - shapely
    - numpy
    - Completed delineate_watersheds.py first
"""

import logging
import sys
import time
from pathlib import Path

import fiona
import numpy as np
import rasterio
import rasterio.mask
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

DEM_MOSAIC      = config.DATA_DEM_MOSAIC / "dem_mosaic.tif"          # Full DEM mosaic
WATERSHEDS_SHP  = config.DATA_SCRATCH_WATERSHEDS / "watersheds.shp"  # Watershed polygons
OUTPUT_DIR      = config.DATA_WATERSHEDS                              # Output folder for clipped DEMs

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


def clip_watershed(
    dem_ds: rasterio.DatasetReader,
    geom: dict,
    watershed_id: int,
    output_dir: Path,
    logger: logging.Logger,
) -> tuple[bool, str]:
    """
    Clips the DEM to a single watershed polygon geometry (GeoJSON-like dict).
    Returns (success, output_path).
    """
    out_path = output_dir / f"watershed_{watershed_id}.tif"

    if out_path.exists():
        return (True, str(out_path))

    try:
        # rasterio.mask.mask expects a list of geometry dicts
        clipped_data, clipped_transform = rasterio.mask.mask(
            dem_ds,
            [geom],
            crop=True,        # Crop to the bounding box of the geometry
            filled=True,      # Fill nodata outside the mask
            nodata=dem_ds.nodata if dem_ds.nodata is not None else np.nan,
        )

        out_meta = dem_ds.meta.copy()
        out_meta.update({
            "driver":    "GTiff",
            "height":    clipped_data.shape[1],
            "width":     clipped_data.shape[2],
            "transform": clipped_transform,
            "compress":  "lzw",
            "tiled":     True,
        })

        with rasterio.open(str(out_path), "w", **out_meta) as dst:
            dst.write(clipped_data)

        return (True, str(out_path))

    except Exception as e:
        logger.error(f"  Failed to clip watershed {watershed_id}: {e}")
        if out_path.exists():
            try:
                out_path.unlink()
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

    # Load watershed features
    with fiona.open(str(watersheds_shp)) as shp:
        # Check that the ID field exists
        if ID_FIELD not in shp.schema["properties"]:
            available = list(shp.schema["properties"].keys())
            logger.error(
                f"ID_FIELD '{ID_FIELD}' not found in shapefile. "
                f"Available fields: {available}"
            )
            sys.exit(1)

        # Read CRS for reprojection check later
        shp_crs = shp.crs
        features = [
            (feat["geometry"], feat["properties"][ID_FIELD])
            for feat in shp
        ]

    total = len(features)
    logger.info(f"Found {total} watersheds")
    logger.info(f"DEM mosaic : {dem_path}")
    logger.info(f"Output dir : {output_dir}")
    logger.info("-" * 60)

    start_time = time.time()
    succeeded  = 0
    failed     = 0
    skipped    = 0

    # Open the DEM once and reuse across all clips for efficiency
    with rasterio.open(str(dem_path)) as dem_ds:

        # Warn if CRS mismatch between shapefile and DEM
        if shp_crs and dem_ds.crs and shp_crs != dem_ds.crs:
            logger.warning(
                f"CRS mismatch: shapefile={shp_crs}, DEM={dem_ds.crs}. "
                "Geometries will be used as-is; reproject your shapefile if "
                "results look incorrect."
            )

        for i, (geom, wid) in enumerate(features, start=1):
            out_path = output_dir / f"watershed_{wid}.tif"

            if out_path.exists():
                skipped += 1
                logger.info(f"[{i:3d}/{total}] SKIP  Watershed {wid} — already exists")
                continue

            logger.info(f"[{i:3d}/{total}] START Watershed {wid}...")
            tile_start = time.time()

            success, result_path = clip_watershed(
                dem_ds, geom, wid, output_dir, logger
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
