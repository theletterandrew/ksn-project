"""
clip_watersheds.py
------------------
Clips the full DEM mosaic and FAC raster to each watershed polygon,
producing individual watershed DEMs and FAC rasters suitable for ksn
analysis.

For each watershed polygon, extracts the corresponding DEM and FAC
extents and saves them as separate GeoTIFF files named by the watershed
ID, e.g.:
    watershed_1.tif       (clipped DEM)
    watershed_1_fac.tif   (clipped FAC, grid-aligned to DEM)

Having per-watershed FAC rasters on an identical grid to their DEM
eliminates the reprojection step in calculate_ksn.py and prevents
spurious stream points caused by grid misalignment.

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
    - Completed delineate_watersheds.py and wbt_hydrology.py first
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

DEM_MOSAIC     = config.DATA_DEM_MOSAIC / "dem_mosaic.tif"           # Full DEM mosaic
FAC_RASTER     = config.DATA_SCRATCH_WBT / "flow_accumulation.tif"  # Full FAC raster
WATERSHEDS_SHP = config.DATA_SCRATCH_WATERSHEDS / "watersheds.shp"  # Watershed polygons
OUTPUT_DIR     = config.DATA_WATERSHEDS                              # Output folder

# Field in watersheds.shp that contains unique watershed IDs
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


def clip_raster(
    ds: rasterio.DatasetReader,
    geom: dict,
    out_path: Path,
    nodata_override=None,
    logger: logging.Logger = None,
) -> tuple[bool, str]:
    """
    Clips an open rasterio dataset to a single watershed polygon geometry.
    The output grid is cropped to the geometry bounding box and masked
    outside the polygon boundary.

    nodata_override: use this nodata value in the output if the source
    dataset has no nodata defined (e.g. some FAC rasters).

    Returns (success, output_path).
    """
    if out_path.exists():
        return (True, str(out_path))

    nodata_val = ds.nodata if ds.nodata is not None else nodata_override

    try:
        clipped_data, clipped_transform = rasterio.mask.mask(
            ds,
            [geom],
            crop=True,
            filled=True,
            nodata=nodata_val if nodata_val is not None else np.nan,
        )

        out_meta = ds.meta.copy()
        out_meta.update({
            "driver":    "GTiff",
            "height":    clipped_data.shape[1],
            "width":     clipped_data.shape[2],
            "transform": clipped_transform,
            "compress":  "lzw",
            "tiled":     True,
        })
        if nodata_val is not None:
            out_meta["nodata"] = nodata_val

        with rasterio.open(str(out_path), "w", **out_meta) as dst:
            dst.write(clipped_data)

        return (True, str(out_path))

    except Exception as e:
        if logger:
            logger.error(f"  Failed to clip {out_path.name}: {e}")
        if out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                pass
        return (False, "")


def main():
    dem_path       = Path(DEM_MOSAIC)
    fac_path       = Path(FAC_RASTER)
    watersheds_shp = Path(WATERSHEDS_SHP)
    output_dir     = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    # Validate inputs
    for label, path in [
        ("DEM mosaic",          dem_path),
        ("FAC raster",          fac_path),
        ("Watersheds shapefile", watersheds_shp),
    ]:
        if not path.exists():
            logger.error(f"{label} not found: {path}")
            sys.exit(1)

    # Load watershed features
    with fiona.open(str(watersheds_shp)) as shp:
        if ID_FIELD not in shp.schema["properties"]:
            available = list(shp.schema["properties"].keys())
            logger.error(
                f"ID_FIELD '{ID_FIELD}' not found in shapefile. "
                f"Available fields: {available}"
            )
            sys.exit(1)

        shp_crs  = shp.crs
        features = [
            (feat["geometry"], feat["properties"][ID_FIELD])
            for feat in shp
        ]

    total = len(features)
    logger.info(f"Found {total} watersheds")
    logger.info(f"DEM mosaic  : {dem_path}")
    logger.info(f"FAC raster  : {fac_path}")
    logger.info(f"Output dir  : {output_dir}")
    logger.info("-" * 60)

    start_time = time.time()
    succeeded  = 0
    failed     = 0
    skipped    = 0

    with rasterio.open(str(dem_path)) as dem_ds, \
         rasterio.open(str(fac_path)) as fac_ds:

        # Warn on CRS mismatches
        for label, ds_crs in [("DEM", dem_ds.crs), ("FAC", fac_ds.crs)]:
            if shp_crs and ds_crs and shp_crs != ds_crs:
                logger.warning(
                    f"CRS mismatch: shapefile={shp_crs}, {label}={ds_crs}. "
                    "Reproject your shapefile if results look incorrect."
                )

        for i, (geom, wid) in enumerate(features, start=1):
            dem_out = output_dir / f"watershed_{wid}.tif"
            fac_out = output_dir / f"watershed_{wid}_fac.tif"

            dem_exists = dem_out.exists()
            fac_exists = fac_out.exists()

            if dem_exists and fac_exists:
                skipped += 1
                logger.info(f"[{i:3d}/{total}] SKIP  Watershed {wid} — both files exist")
                continue

            logger.info(f"[{i:3d}/{total}] START Watershed {wid}...")
            tile_start = time.time()

            # Clip DEM
            dem_ok = True
            if not dem_exists:
                dem_ok, _ = clip_raster(dem_ds, geom, dem_out, logger=logger)
                if not dem_ok:
                    logger.error(f"  DEM clip failed for watershed {wid}")

            # Clip FAC — use -9999 as nodata if the FAC has none defined
            fac_ok = True
            if not fac_exists:
                fac_ok, _ = clip_raster(
                    fac_ds, geom, fac_out,
                    nodata_override=-9999,
                    logger=logger,
                )
                if not fac_ok:
                    logger.error(f"  FAC clip failed for watershed {wid}")

            if dem_ok and fac_ok:
                succeeded += 1
                tile_time = time.time() - tile_start
                dem_mb    = dem_out.stat().st_size / 1024 / 1024
                fac_mb    = fac_out.stat().st_size / 1024 / 1024
                elapsed   = time.time() - start_time
                rate      = i / elapsed
                eta_min   = (total - i) / rate / 60 if rate > 0 else 0
                logger.info(
                    f"[{i:3d}/{total}] OK    Watershed {wid}  |  "
                    f"DEM {dem_mb:.1f} MB  FAC {fac_mb:.1f} MB  |  "
                    f"{tile_time:.1f}s  |  ETA {eta_min:.1f} min"
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
    logger.info("Watershed DEMs and FAC rasters ready for calculate_ksn.py.")


if __name__ == "__main__":
    main()
