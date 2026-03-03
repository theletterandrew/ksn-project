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
    --> extract_longest_branches.py   (this script)
    calculate_ksn.py

Inputs (produced by clip_watersheds.py):
    DATA_WATERSHEDS / watershed_{wid}_fdr.tif   — clipped FDR per watershed

Inputs (produced by delineate_watersheds.py):
    DATA_SCRATCH_WATERSHEDS / watersheds.tif    — multi-watershed label raster

Output:
    DATA_STREAMS / longest_branches.gpkg        — one LineString per watershed,
                                                  ordered mouth->headwater,
                                                  with watershed_id attribute

USAGE:
    1. Install dependencies:
       pip install rasterio numpy fiona shapely

    2. Edit the paths in the CONFIG section below if needed.

    3. Run:
       python extract_longest_branches.py

Requirements:
    - rasterio
    - numpy
    - fiona
    - shapely
    - WhiteboxTools v2.4.0+ executable configured via config.WBT_EXE
    - Completed clip_watersheds.py first
"""

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
import fiona
import fiona.crs
from shapely.geometry import LineString, mapping, shape as shapely_shape

# Calculate the path to the project root (one level up from scripts/)
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

# =============================================================================
# CONFIG — Edit these before running
# =============================================================================

WATERSHEDS_DIR  = config.DATA_WATERSHEDS          # clipped per-watershed rasters
SCRATCH_DIR     = config.DATA_SCRATCH_WATERSHEDS  # watersheds.tif lives here
OUTPUT_DIR      = config.DATA_STREAMS             # longest_branches.gpkg goes here
WBT_EXE         = config.WBT_EXE

OUTPUT_FILE     = "longest_branches.gpkg"

# Field in watersheds.shp that contains unique watershed IDs (must match
# delineate_watersheds.py / clip_watersheds.py).
ID_FIELD        = "gridcode"

# Minimum flowpath length in metres. Watersheds whose longest path is shorter
# than this are skipped (usually tiny edge-clipped slivers).
MIN_LENGTH_M    = getattr(config, "MIN_STREAM_LENGTH_M", 100.0)

# =============================================================================
# END CONFIG — No edits needed below this line
# =============================================================================


def setup_logging(output_dir: Path) -> logging.Logger:
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


def run_wbt(tool: str, args: dict, logger: logging.Logger, timeout: int = 300) -> bool:
    """Run a WhiteboxTools command via subprocess, streaming output in real time."""
    cmd = [str(WBT_EXE), f"--run={tool}"]
    for key, val in args.items():
        cmd.append(f"--{key}={val}")
    logger.info(f"  WBT {tool}: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

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


def raster_to_linestring(
    flowpath_tif: Path,
    transform,
    crs,
    logger: logging.Logger,
) -> LineString | None:
    """
    Convert a LongestFlowpath output raster to a single ordered LineString.

    LongestFlowpath writes a raster where flowpath cells have value 1 and
    everything else is nodata. We extract those cells, sort them from mouth
    (highest FAC proxy = highest row index in a south-draining DEM) to
    headwater by chaining each cell to its 8-connected neighbour, and
    convert to map coordinates.

    Sorting strategy: start from the cell with the largest row index
    (furthest downstream in typical north-up rasters) and walk to the
    end of the chain using 8-connectivity, ensuring a continuous line
    rather than a spatially scrambled point cloud.
    """
    with rasterio.open(str(flowpath_tif)) as src:
        arr    = src.read(1)
        nd     = src.nodata
        trans  = src.transform

    # Identify flowpath cells
    if nd is not None:
        mask = (arr != nd) & (arr > 0)
    else:
        mask = arr > 0

    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None

    pixel_set = set(zip(rows.tolist(), cols.tolist()))

    # Walk the chain from the southernmost (highest row) cell
    start = max(pixel_set, key=lambda p: p[0])
    chain = [start]
    visited = {start}

    while True:
        r, c = chain[-1]
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

    # Convert pixel centres to map coordinates
    coords = [
        (trans.c + (c + 0.5) * trans.a,
         trans.f + (r + 0.5) * trans.e)
        for r, c in chain
    ]
    return LineString(coords)


def extract_watershed_mask(
    watersheds_tif: Path,
    wid: int,
) -> Path:
    """
    Write a temporary single-watershed mask raster (value=1 where
    watersheds_tif == wid, nodata elsewhere) for use as the LongestFlowpath
    watershed input.

    Returns the path to the temporary raster.
    """
    out_path = watersheds_tif.parent / f"_ws_mask_{wid}.tif"
    if out_path.exists():
        return out_path

    with rasterio.open(str(watersheds_tif)) as src:
        arr  = src.read(1)
        meta = src.meta.copy()
        nd   = src.nodata if src.nodata is not None else 0

    mask = np.where(arr == wid, 1, nd).astype(np.int32)
    meta.update(dtype=rasterio.int32, nodata=int(nd))

    with rasterio.open(str(out_path), "w", **meta) as dst:
        dst.write(mask, 1)

    return out_path


def main():
    watersheds_dir  = Path(WATERSHEDS_DIR)
    scratch_dir     = Path(SCRATCH_DIR)
    output_dir      = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    watersheds_tif = scratch_dir / "watersheds.tif"
    out_gpkg       = output_dir  / OUTPUT_FILE

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not watersheds_tif.exists():
        logger.error(f"watersheds.tif not found: {watersheds_tif}")
        logger.error("Run delineate_watersheds.py first.")
        sys.exit(1)

    # Discover all clipped FDR rasters produced by clip_watersheds.py
    fdr_files = sorted(watersheds_dir.glob("watershed_*_fdr.tif"))
    if not fdr_files:
        logger.error(f"No watershed_*_fdr.tif files found in {watersheds_dir}")
        logger.error("Run clip_watersheds.py first.")
        sys.exit(1)

    # Parse watershed IDs from filenames
    watersheds = []
    for fdr_path in fdr_files:
        # filename: watershed_{wid}_fdr.tif
        parts = fdr_path.stem.split("_")  # ['watershed', '{wid}', 'fdr']
        try:
            wid = int(parts[1])
            watersheds.append((wid, fdr_path))
        except (IndexError, ValueError):
            logger.warning(f"  Could not parse watershed ID from {fdr_path.name}, skipping.")

    if not watersheds:
        logger.error("No valid watershed FDR files found.")
        sys.exit(1)

    logger.info(f"Watersheds found : {len(watersheds)}")
    logger.info(f"watersheds.tif   : {watersheds_tif}")
    logger.info(f"Output           : {out_gpkg}")
    logger.info(f"Min length       : {MIN_LENGTH_M} m")
    logger.info("-" * 60)

    # ------------------------------------------------------------------
    # Read CRS from the first FDR raster (all share the same CRS)
    # ------------------------------------------------------------------
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
    succeeded  = 0
    skipped    = 0
    failed     = 0

    with fiona.open(
        str(out_gpkg), mode="w", driver="GPKG",
        schema=schema, crs=out_crs, layer="longest_branches",
    ) as dst:

        for wid, fdr_path in watersheds:
            logger.info(f"Processing watershed {wid}  ({fdr_path.name})...")

            # Temp paths for this watershed
            ws_mask_tif  = scratch_dir / f"_ws_mask_{wid}.tif"
            flowpath_tif = scratch_dir / f"_flowpath_{wid}.tif"

            try:
                # ------------------------------------------------------
                # Step 1: Extract single-watershed mask raster
                # ------------------------------------------------------
                ws_mask_tif = extract_watershed_mask(watersheds_tif, wid)

                # ------------------------------------------------------
                # Step 2: Run WBT LongestFlowpath
                # ------------------------------------------------------
                ok = run_wbt("LongestFlowpath", {
                    "d8_pntr"   : str(fdr_path),
                    "watersheds": str(ws_mask_tif),
                    "output"    : str(flowpath_tif),
                }, logger)

                if not ok or not flowpath_tif.exists():
                    logger.warning(f"  LongestFlowpath failed for watershed {wid}, skipping.")
                    failed += 1
                    continue

                # ------------------------------------------------------
                # Step 3: Vectorise flowpath raster -> LineString
                # ------------------------------------------------------
                with rasterio.open(str(fdr_path)) as src:
                    transform = src.transform

                line = raster_to_linestring(flowpath_tif, transform, crs, logger)

                if line is None:
                    logger.warning(f"  No flowpath pixels found for watershed {wid}, skipping.")
                    skipped += 1
                    continue

                if line.length < MIN_LENGTH_M:
                    logger.info(
                        f"  Watershed {wid} flowpath {line.length:.1f} m "
                        f"< MIN_LENGTH_M ({MIN_LENGTH_M} m), skipping."
                    )
                    skipped += 1
                    continue

                # ------------------------------------------------------
                # Step 4: Write to GeoPackage
                # ------------------------------------------------------
                dst.write({
                    "geometry": mapping(line),
                    "properties": {
                        "watershed_id": wid,
                        "length_m":     round(line.length, 2),
                        "n_vertices":   len(line.coords),
                    },
                })

                logger.info(
                    f"  OK  watershed {wid}  |  "
                    f"{line.length / 1000:.2f} km  |  "
                    f"{len(line.coords)} vertices"
                )
                succeeded += 1

            finally:
                # Clean up temporary rasters regardless of success/failure
                for tmp in [ws_mask_tif, flowpath_tif]:
                    try:
                        Path(tmp).unlink(missing_ok=True)
                    except Exception:
                        pass

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info(f"  Output     : {out_gpkg}")
    logger.info(f"  Succeeded  : {succeeded}")
    logger.info(f"  Skipped    : {skipped}  (too short or no pixels)")
    logger.info(f"  Failed     : {failed}   (WBT error)")
    logger.info(f"  Total time : {elapsed / 60:.1f} minutes")
    logger.info("")
    logger.info("Load longest_branches.gpkg in ArcGIS Pro / QGIS to verify.")
    logger.info("Each feature is a LineString ordered mouth->headwater.")


if __name__ == "__main__":
    main()
