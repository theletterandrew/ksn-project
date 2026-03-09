"""
02_delineate_and_clip_basins.py

Takes the outputs of the hydrological conditioning script and:
  1. Delineates drainage basins using WhiteboxTools (via subprocess)
  2. Converts the basin raster to vector polygons
  3. Clips the breached DEM to each basin and saves as individual GeoTIFFs
  4. Generates a hillshade TIF for each clipped basin DEM

Inputs (from conditioning script):
  - Filled/breached DEM  : <CONDITIONING_DIR>/dem_breached.tif
  - D8 flow pointer      : <CONDITIONING_DIR>/d8_pointer.tif

Outputs (in <CONDITIONING_DIR>/basins/):
  - basins_raster.tif    : Integer raster, one unique value per basin
  - basins_polygons.shp  : Vector polygons of each basin
  - basin_<ID>/dem.tif   : Clipped breached DEM for each qualifying basin
  - basin_<ID>/hillshade.tif : Hillshade for each qualifying basin

Dependencies:
  pip install rasterio numpy geopandas shapely
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
import config
from config import WBT_EXE

# ─── USER CONFIG ──────────────────────────────────────────────────────────────

# Directory containing outputs from your conditioning script
CONDITIONING_DIR = config.DATA_SCRATCH_WBT

# Minimum basin area to export (km²) — filters out small edge-draining slivers
MIN_BASIN_AREA_KM2 = config.MIN_BASIN_AREA_KM2

# UTM CRS for area calculation — UTM Zone 11N is correct for San Bernardino Mountains
AREA_CRS = "EPSG:32611"

# ─── HILLSHADE CONFIG ─────────────────────────────────────────────────────────

# Sun azimuth in degrees (0–360, clockwise from north; 315 = northwest is standard)
HILLSHADE_AZIMUTH = 315.0

# Sun altitude angle in degrees above the horizon (0–90; 45 is a common default)
HILLSHADE_ALTITUDE = 45.0

# ─── DERIVED PATHS ────────────────────────────────────────────────────────────

BREACHED_DEM  = os.path.join(CONDITIONING_DIR, "dem_filled.tif")
D8_POINTER    = os.path.join(CONDITIONING_DIR, "flow_direction.tif")

OUTPUT_DIR    = config.DATA_BASINS
BASINS_RASTER = os.path.join(OUTPUT_DIR, "basins_raster.tif")
BASINS_VECTOR = os.path.join(OUTPUT_DIR, "basins_polygons.shp")

# ─── LOGGING ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(CONDITIONING_DIR, "delineate_basins.log"))
    ]
)
logger = logging.getLogger(__name__)

# ─── WBT HELPER (matches your conditioning script) ────────────────────────────

def run_wbt(tool: str, args: dict, logger: logging.Logger) -> bool:
    """
    Runs a WhiteboxTools command. Returns True on success.
    args is a dict of parameter name -> value.
    """
    cmd = [str(WBT_EXE), f"--run={tool}"]
    for key, val in args.items():
        cmd.append(f"--{key}={val}")
    logger.info(f"Running: {tool}")
    logger.info(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.stdout:
            for line in result.stdout.strip().splitlines():
                logger.info(f"  WBT: {line}")
        if result.stderr:
            for line in result.stderr.strip().splitlines():
                logger.warning(f"  WBT ERR: {line}")
        if result.returncode != 0:
            logger.error(f"{tool} failed with return code {result.returncode}")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to run {tool}: {e}")
        return False

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Delineate basins ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 1: Delineating drainage basins")
    logger.info("=" * 60)

    success = run_wbt("Basins", {
        "d8_pntr": D8_POINTER,
        "output":  BASINS_RASTER,
    }, logger)

    if not success:
        logger.error("Basins failed — aborting.")
        sys.exit(1)

    # ── Step 2: Vectorize basins ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 2: Converting basin raster to vector polygons")
    logger.info("=" * 60)

    success = run_wbt("RasterToVectorPolygons", {
        "i":      BASINS_RASTER,
        "output": BASINS_VECTOR,
    }, logger)

    if not success:
        logger.error("RasterToVectorPolygons failed — aborting.")
        sys.exit(1)

    # ── Step 3: Filter small basins and clip DEM ───────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 3: Filtering basins and clipping DEM")
    logger.info("=" * 60)

    basins_gdf = gpd.read_file(BASINS_VECTOR)
    logger.info(f"Total basins delineated: {len(basins_gdf)}")

    # Compute area in km² using equal-area projection
    crs = basins_gdf.crs
    if crs and crs.is_geographic:
        basins_projected = basins_gdf.to_crs(AREA_CRS)
    else:
        basins_projected = basins_gdf

    basins_gdf["area_km2"] = basins_projected.geometry.area / 1e6

    logger.info("Basin area summary (km²):")
    for line in str(basins_gdf["area_km2"].describe().round(2)).splitlines():
        logger.info(f"  {line}")

    large_basins = basins_gdf[basins_gdf["area_km2"] >= MIN_BASIN_AREA_KM2].copy()
    logger.info(f"Basins >= {MIN_BASIN_AREA_KM2} km²: {len(large_basins)}")
    logger.info(f"Skipped (too small): {len(basins_gdf) - len(large_basins)}")

    if len(large_basins) == 0:
        logger.error(
            f"No basins found above {MIN_BASIN_AREA_KM2} km². "
            "Try lowering MIN_BASIN_AREA_KM2."
        )
        sys.exit(1)

    large_basins.to_file(BASINS_VECTOR, driver="ESRI Shapefile")
    logger.info(f"Saved filtered basin polygons ({len(large_basins)} basins) to {BASINS_VECTOR}")

    # ── Steps 3 & 4: Clip DEM and generate hillshade per basin ────────────────
    logger.info("=" * 60)
    logger.info("Steps 3 & 4: Clipping DEM and generating hillshades")
    logger.info("=" * 60)

    hillshade_ok = 0
    hillshade_fail = 0

    with rasterio.open(BREACHED_DEM) as src:
        dem_crs = src.crs

        if large_basins.crs != dem_crs:
            large_basins = large_basins.to_crs(dem_crs)

        for _, row in large_basins.iterrows():
            basin_id  = int(row.get("VALUE", row.name))
            area_km2  = row["area_km2"]
            basin_dir = os.path.join(OUTPUT_DIR, f"basin_{basin_id:04d}")
            os.makedirs(basin_dir, exist_ok=True)
            dem_path       = os.path.join(basin_dir, "dem.tif")
            hillshade_path = os.path.join(basin_dir, "hillshade.tif")

            # ── Step 3: Clip DEM ───────────────────────────────────────────────
            try:
                clipped, transform = rio_mask(
                    src, [row.geometry], crop=True, nodata=src.nodata
                )

                out_meta = src.meta.copy()
                out_meta.update({
                    "driver":    "GTiff",
                    "height":    clipped.shape[1],
                    "width":     clipped.shape[2],
                    "transform": transform,
                    "compress":  "lzw"
                })

                with rasterio.open(dem_path, "w", **out_meta) as dst:
                    dst.write(clipped)

                logger.info(f"  Wrote basin_{basin_id:04d}/dem.tif  ({area_km2:.1f} km²)")

            except Exception as e:
                logger.error(f"  Could not clip basin {basin_id}: {e}")
                continue  # Skip hillshade if DEM clip failed

            # ── Step 4: Generate hillshade from clipped DEM ────────────────────
            success = run_wbt("Hillshade", {
                "dem":      dem_path,
                "output":   hillshade_path,
                "azimuth":  HILLSHADE_AZIMUTH,
                "altitude": HILLSHADE_ALTITUDE,
            }, logger)

            if success:
                logger.info(f"  Wrote basin_{basin_id:04d}/hillshade.tif")
                hillshade_ok += 1
            else:
                logger.error(f"  Hillshade failed for basin_{basin_id:04d}")
                hillshade_fail += 1

    logger.info("=" * 60)
    logger.info("Done.")
    logger.info(f"  Basin raster   : {BASINS_RASTER}")
    logger.info(f"  Basin polygons : {BASINS_VECTOR}")
    logger.info(f"  Clipped DEMs   : {OUTPUT_DIR}/basin_XXXX/dem.tif")
    logger.info(f"  Hillshades     : {OUTPUT_DIR}/basin_XXXX/hillshade.tif")
    logger.info(f"  Hillshades OK  : {hillshade_ok}  |  Failed: {hillshade_fail}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
