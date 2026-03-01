"""
clip_watersheds.py
------------------
Clips the full DEM mosaic, FAC, and FDR rasters to each watershed polygon,
producing individual per-watershed rasters suitable for ksn analysis.

For each watershed polygon, produces:
    watershed_1.tif       (clipped DEM)
    watershed_1_fac.tif   (clipped FAC, grid-aligned to DEM)
    watershed_1_fdr.tif   (clipped FDR, grid-aligned to DEM)

All three rasters share an identical grid, eliminating any reprojection
in calculate_ksn.py and preventing spurious stream points from grid
misalignment.

After clipping, the FAC is masked to the DEM's valid extent so that cells
outside the watershed polygon (but inside the bounding box) do not retain
non-zero accumulation values that would produce off-channel ksn points.

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

# Calculate the path to the project root (one level up from scripts/)
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

# =============================================================================
# CONFIG — Edit these before running
# =============================================================================

DEM_MOSAIC     = config.DATA_DEM_MOSAIC / "dem_mosaic.tif"
FAC_RASTER     = config.DATA_SCRATCH_WBT / "flow_accumulation.tif"
FDR_RASTER     = config.DATA_SCRATCH_WBT / "flow_direction.tif"
WATERSHEDS_SHP = config.DATA_SCRATCH_WATERSHEDS / "watersheds.shp"
OUTPUT_DIR     = config.DATA_WATERSHEDS

# Field in watersheds.shp that contains unique watershed IDs
ID_FIELD = "gridcode"

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
            logging.StreamHandler(sys.stdout),
        ],
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
    Crops to the bounding box and masks cells outside the polygon.

    nodata_override: use this nodata value if the source has none defined.
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
    fdr_path       = Path(FDR_RASTER)
    watersheds_shp = Path(WATERSHEDS_SHP)
    output_dir     = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    # Validate inputs
    for label, path in [
        ("DEM mosaic",           dem_path),
        ("FAC raster",           fac_path),
        ("FDR raster",           fdr_path),
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
    logger.info(f"DEM mosaic   : {dem_path}")
    logger.info(f"FAC raster   : {fac_path}")
    logger.info(f"FDR raster   : {fdr_path}")
    logger.info(f"Output dir   : {output_dir}")
    logger.info("-" * 60)

    start_time = time.time()
    succeeded  = 0
    failed     = 0
    skipped    = 0

    with rasterio.open(str(dem_path)) as dem_ds, \
         rasterio.open(str(fac_path)) as fac_ds, \
         rasterio.open(str(fdr_path)) as fdr_ds:

        # Warn on CRS mismatches
        for label, ds_crs in [
            ("DEM", dem_ds.crs),
            ("FAC", fac_ds.crs),
            ("FDR", fdr_ds.crs),
        ]:
            if shp_crs and ds_crs and shp_crs != ds_crs:
                logger.warning(
                    f"CRS mismatch: shapefile={shp_crs}, {label}={ds_crs}. "
                    "Reproject your shapefile if results look incorrect."
                )

        for i, (geom, wid) in enumerate(features, start=1):
            dem_out = output_dir / f"watershed_{wid}.tif"
            fac_out = output_dir / f"watershed_{wid}_fac.tif"
            fdr_out = output_dir / f"watershed_{wid}_fdr.tif"

            if dem_out.exists() and fac_out.exists() and fdr_out.exists():
                skipped += 1
                logger.info(
                    f"[{i:3d}/{total}] SKIP  Watershed {wid} — all files exist"
                )
                continue

            logger.info(f"[{i:3d}/{total}] START Watershed {wid}...")
            tile_start = time.time()

            # Clip DEM
            dem_ok = True
            if not dem_out.exists():
                dem_ok, _ = clip_raster(dem_ds, geom, dem_out, logger=logger)
                if not dem_ok:
                    logger.error(f"  DEM clip failed for watershed {wid}")

            # Clip FAC
            fac_ok = True
            if not fac_out.exists():
                fac_ok, _ = clip_raster(
                    fac_ds, geom, fac_out,
                    nodata_override=-9999,
                    logger=logger,
                )
                if not fac_ok:
                    logger.error(f"  FAC clip failed for watershed {wid}")

            # Clip FDR
            fdr_ok = True
            if not fdr_out.exists():
                fdr_ok, _ = clip_raster(
                    fdr_ds, geom, fdr_out,
                    nodata_override=0,
                    logger=logger,
                )
                if not fdr_ok:
                    logger.error(f"  FDR clip failed for watershed {wid}")

            if dem_ok and fac_ok and fdr_ok:
                # ------------------------------------------------------
                # Mask the clipped FAC to the DEM's valid extent.
                # Cells outside the watershed polygon but inside the
                # bounding box retain non-zero FAC values from the mosaic;
                # zeroing them prevents off-channel ksn points.
                # ------------------------------------------------------
                with rasterio.open(str(dem_out)) as dem_src:
                    dem_arr    = dem_src.read(1)
                    dem_nodata = dem_src.nodata

                with rasterio.open(str(fac_out)) as fac_src:
                    fac_meta = fac_src.meta.copy()
                    fac_arr  = fac_src.read(1)
                    fac_nd   = fac_src.nodata

                if dem_nodata is not None:
                    fill = fac_nd if fac_nd is not None else -9999
                    fac_arr[dem_arr == dem_nodata] = fill

                with rasterio.open(str(fac_out), "w", **fac_meta) as dst:
                    dst.write(fac_arr, 1)

                succeeded += 1
                tile_time = time.time() - tile_start
                dem_mb    = dem_out.stat().st_size / 1024 / 1024
                fac_mb    = fac_out.stat().st_size / 1024 / 1024
                fdr_mb    = fdr_out.stat().st_size / 1024 / 1024
                elapsed   = time.time() - start_time
                rate      = i / elapsed
                eta_min   = (total - i) / rate / 60 if rate > 0 else 0
                logger.info(
                    f"[{i:3d}/{total}] OK    Watershed {wid}  |  "
                    f"DEM {dem_mb:.1f} MB  "
                    f"FAC {fac_mb:.1f} MB  "
                    f"FDR {fdr_mb:.1f} MB  |  "
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
    logger.info("Watershed DEMs, FAC, and FDR rasters ready for calculate_ksn.py.")


if __name__ == "__main__":
    main()
