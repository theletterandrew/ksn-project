"""
extract_longest_branches.py
---------------------------
Extracts the longest mouth->headwater flowpath for every watershed using
WhiteboxTools' LongestFlowpath tool, then vectorises the results into a
single GeoPackage with one LineString per watershed.

Pipeline position:
    wbt_hydrology.py
    stream_extraction_wbt.py
    delineate_watersheds.py
    clip_watersheds.py
    --> extract_longest_branches.py
    calculate_ksn.py

Inputs (produced by clip_watersheds.py):
    DATA_WATERSHEDS / watershed_{wid}_fdr.tif

Inputs (produced by delineate_watersheds.py):
    DATA_SCRATCH_WATERSHEDS / watersheds.shp

Output:
    DATA_STREAMS / longest_branches.gpkg
"""

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rasterio
import fiona
from shapely.geometry import LineString, mapping

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

# =============================================================================
# CONFIG
# =============================================================================

WATERSHEDS_DIR = config.DATA_WATERSHEDS
SCRATCH_DIR    = config.DATA_SCRATCH_WATERSHEDS
OUTPUT_DIR     = config.DATA_STREAMS
WBT_EXE        = config.WBT_EXE

WATERSHEDS_SHP = config.DATA_SCRATCH_WATERSHEDS / "watersheds.shp"
OUTPUT_FILE    = "longest_branches.gpkg"
ID_FIELD       = "gridcode"
MIN_LENGTH_M   = getattr(config, "MIN_STREAM_LENGTH_M", 100.0)

# =============================================================================


def setup_logging(output_dir):
    log_path = output_dir / "extract_longest_branches.log"
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


def run_wbt(tool, args, logger, timeout=300):
    cmd = [str(WBT_EXE), f"--run={tool}"]
    for key, val in args.items():
        cmd.append(f"--{key}={val}")
    logger.info(f"  WBT {tool}: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        def _stream(pipe, log_fn):
            for line in iter(pipe.readline, ""):
                line = line.rstrip()
                if line:
                    log_fn(f"    WBT: {line}")
            pipe.close()

        t_out = threading.Thread(target=_stream, args=(process.stdout, logger.info), daemon=True)
        t_err = threading.Thread(target=_stream, args=(process.stderr, logger.warning), daemon=True)
        t_out.start()
        t_err.start()

        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            logger.error(f"  {tool} killed after {timeout}s timeout.")
            return False

        t_out.join(timeout=5)
        t_err.join(timeout=5)

        if process.returncode != 0:
            logger.error(f"  {tool} failed (return code {process.returncode})")
            return False
        return True

    except Exception as e:
        logger.error(f"  Failed to run {tool}: {e}")
        return False


def make_watershed_mask(fdr_path, out_path):
    """
    Write a mask raster aligned to the clipped FDR where all valid cells = 1.
    LongestFlowpath uses this to define the watershed extent; since the FDR
    is already clipped to the watershed boundary, this is sufficient.
    """
    with rasterio.open(str(fdr_path)) as src:
        arr  = src.read(1)
        meta = src.meta.copy()
        nd   = src.nodata

    NODATA_VAL = 0
    mask = np.where(arr != nd, 1, NODATA_VAL).astype(np.int32) if nd is not None \
           else np.ones(arr.shape, dtype=np.int32)

    meta.update(dtype=rasterio.int32, nodata=NODATA_VAL)
    with rasterio.open(str(out_path), "w", **meta) as dst:
        dst.write(mask, 1)


def raster_to_linestring(flowpath_tif):
    """
    Walk the 8-connected flowpath pixel chain from the southernmost cell
    (mouth) to the headwater and return an ordered LineString.
    """
    with rasterio.open(str(flowpath_tif)) as src:
        arr   = src.read(1)
        nd    = src.nodata
        trans = src.transform

    mask = (arr != nd) & (arr > 0) if nd is not None else arr > 0
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None

    pixel_set = set(zip(rows.tolist(), cols.tolist()))
    start     = max(pixel_set, key=lambda p: p[0])
    chain     = [start]
    visited   = {start}

    while True:
        r, c  = chain[-1]
        found = None
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in pixel_set and nb not in visited:
                    found = nb
                    break
            if found:
                break
        if found is None:
            break
        chain.append(found)
        visited.add(found)

    if len(chain) < 2:
        return None

    coords = [
        (trans.c + (c + 0.5) * trans.a,
         trans.f + (r + 0.5) * trans.e)
        for r, c in chain
    ]
    return LineString(coords)


def main():
    watersheds_dir = Path(WATERSHEDS_DIR)
    scratch_dir    = Path(SCRATCH_DIR)
    output_dir     = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    watersheds_shp = Path(WATERSHEDS_SHP)
    out_gpkg       = output_dir / OUTPUT_FILE

    if not watersheds_shp.exists():
        logger.error(f"Watersheds shapefile not found: {watersheds_shp}")
        logger.error("Run delineate_watersheds.py first.")
        sys.exit(1)

    with fiona.open(str(watersheds_shp)) as shp:
        if ID_FIELD not in shp.schema["properties"]:
            logger.error(f"ID_FIELD '{ID_FIELD}' not in shapefile. Available: {list(shp.schema['properties'].keys())}")
            sys.exit(1)
        wids = [int(feat["properties"][ID_FIELD]) for feat in shp]

    watersheds = []
    for wid in sorted(wids):
        fdr_path = watersheds_dir / f"watershed_{wid}_fdr.tif"
        if fdr_path.exists():
            watersheds.append((wid, fdr_path))
        else:
            logger.warning(f"  FDR raster not found for watershed {wid}, skipping.")

    if not watersheds:
        logger.error(f"No watershed_{{wid}}_fdr.tif files found in {watersheds_dir}.")
        sys.exit(1)

    logger.info(f"Watersheds to process : {len(watersheds)}")
    logger.info(f"Output                : {out_gpkg}")
    logger.info(f"Min length            : {MIN_LENGTH_M} m")
    logger.info("-" * 60)

    with rasterio.open(str(watersheds[0][1])) as src:
        crs = src.crs
    out_crs = crs.to_wkt() if crs else None

    schema = {
        "geometry": "LineString",
        "properties": {
            "watershed_id": "int",
            "length_m":     "float",
            "n_vertices":   "int",
        },
    }

    if out_gpkg.exists():
        out_gpkg.unlink()

    start_time = time.time()
    succeeded = skipped = failed = 0

    with fiona.open(str(out_gpkg), mode="w", driver="GPKG",
                    schema=schema, crs=out_crs, layer="longest_branches") as dst:

        for wid, fdr_path in watersheds:
            logger.info(f"Processing watershed {wid}...")
            ws_mask_tif  = scratch_dir / f"_ws_mask_{wid}.tif"
            flowpath_tif = scratch_dir / f"_flowpath_{wid}.tif"

            try:
                make_watershed_mask(fdr_path, ws_mask_tif)

                ok = run_wbt("LongestFlowpath", {
                    "d8_pntr"   : str(fdr_path),
                    "watersheds": str(ws_mask_tif),
                    "output"    : str(flowpath_tif),
                }, logger)

                if not ok or not flowpath_tif.exists():
                    logger.warning(f"  LongestFlowpath failed for watershed {wid}, skipping.")
                    failed += 1
                    continue

                line = raster_to_linestring(flowpath_tif)

                if line is None:
                    logger.warning(f"  No flowpath pixels for watershed {wid}, skipping.")
                    skipped += 1
                    continue

                if line.length < MIN_LENGTH_M:
                    logger.info(f"  Watershed {wid}: {line.length:.1f} m < {MIN_LENGTH_M} m, skipping.")
                    skipped += 1
                    continue

                dst.write({
                    "geometry": mapping(line),
                    "properties": {
                        "watershed_id": wid,
                        "length_m":     round(line.length, 2),
                        "n_vertices":   len(line.coords),
                    },
                })
                logger.info(f"  OK  watershed {wid}  |  {line.length / 1000:.2f} km  |  {len(line.coords)} vertices")
                succeeded += 1

            finally:
                for tmp in [ws_mask_tif, flowpath_tif]:
                    Path(tmp).unlink(missing_ok=True)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info(f"  Output     : {out_gpkg}")
    logger.info(f"  Succeeded  : {succeeded}")
    logger.info(f"  Skipped    : {skipped}  (too short or no pixels)")
    logger.info(f"  Failed     : {failed}  (WBT error)")
    logger.info(f"  Total time : {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()