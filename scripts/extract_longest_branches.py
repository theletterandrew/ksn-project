"""
extract_longest_branches.py
---------------------------
Extracts the longest mouth->headwater flowpath for every watershed using
a pure-numpy D8 implementation, then writes the results to a single
GeoPackage with one LineString per watershed.

Pipeline position:
    wbt_hydrology.py
    stream_extraction_wbt.py
    delineate_watersheds.py
    clip_watersheds.py
    --> extract_longest_branches.py
    calculate_ksn.py

Inputs (produced by clip_watersheds.py):
    DATA_WATERSHEDS / watershed_{wid}_fdr.tif
    DATA_WATERSHEDS / watershed_{wid}_fac.tif

Inputs (produced by delineate_watersheds.py):
    DATA_SCRATCH_WATERSHEDS / watersheds.shp

Output:
    DATA_STREAMS / longest_branches.gpkg

Algorithm
---------
For each watershed:
  1. Find the outlet cell (highest FAC value).
  2. BFS upstream via D8 pointers, accumulating Euclidean distance
     from the outlet to every reachable cell.
  3. The cell with the greatest distance is the headwater.
  4. Traceback downstream from headwater to outlet via D8 pointers.
  5. Convert pixel chain to a map-coordinate LineString.
"""

import logging
import sys
import time
from collections import deque
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

WATERSHEDS_SHP = config.DATA_SCRATCH_WATERSHEDS / "watersheds.shp"
OUTPUT_FILE    = "longest_branches.gpkg"
ID_FIELD       = "gridcode"
MIN_LENGTH_M   = getattr(config, "MIN_STREAM_LENGTH_M", 100.0)

# =============================================================================

# WBT D8 pointer: value -> (row_offset, col_offset)
D8_OFFSETS = {
    1:   ( 0,  1),
    2:   ( 1,  1),
    4:   ( 1,  0),
    8:   ( 1, -1),
    16:  ( 0, -1),
    32:  (-1, -1),
    64:  (-1,  0),
    128: (-1,  1),
}

# Reverse map: (dr, dc) -> D8 value pointing IN that direction
D8_REVERSE = {v: k for k, v in D8_OFFSETS.items()}

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


def longest_flowpath(
    fdr_arr: np.ndarray,
    fac_arr: np.ndarray,
    fdr_nodata,
    fac_nodata,
    cell_size: float,
) -> list | None:
    """
    Find the longest D8 flowpath in a watershed raster.

    Returns an ordered list of (row, col) pixel coordinates from mouth
    (highest FAC = outlet) to headwater (end of longest path), or None
    if no valid path can be found.
    """
    nrows, ncols = fdr_arr.shape

    # ------------------------------------------------------------------
    # Step 1: Find outlet (highest FAC cell, ignoring border pixels)
    # ------------------------------------------------------------------
    BORDER = 3  # cells to ignore on each edge

    fac_valid = fac_arr.copy().astype(np.float64)
    if fac_nodata is not None:
        fac_valid[fac_arr == fac_nodata] = -1.0
    if fdr_nodata is not None:
        fac_valid[fdr_arr == int(fdr_nodata)] = -1.0

    # Blank border cells to suppress edge accumulation artifacts
    fac_valid[:BORDER,  :] = -1.0
    fac_valid[-BORDER:, :] = -1.0
    fac_valid[:,  :BORDER] = -1.0
    fac_valid[:, -BORDER:] = -1.0

    outlet_flat         = int(np.argmax(fac_valid))
    outlet_r, outlet_c  = divmod(outlet_flat, ncols)

    if fac_valid[outlet_r, outlet_c] <= 0:
        return None

    # ------------------------------------------------------------------
    # Step 2: BFS upstream — accumulate distance from outlet
    # ------------------------------------------------------------------
    # For each neighbour direction, pre-compute which FDR value means
    # that neighbour drains INTO the current cell, and the step distance.
    UPSTREAM_CHECKS = []
    for (dr, dc) in D8_OFFSETS.values():
        reverse_offset = (-dr, -dc)
        if reverse_offset in D8_REVERSE:
            expected_fdr = D8_REVERSE[reverse_offset]
            step_dist    = cell_size * (2 ** 0.5 if dr != 0 and dc != 0 else 1.0)
            UPSTREAM_CHECKS.append((dr, dc, expected_fdr, step_dist))

    dist = np.full((nrows, ncols), -1.0, dtype=np.float64)
    dist[outlet_r, outlet_c] = 0.0

    queue = deque()
    queue.append((outlet_r, outlet_c))

    while queue:
        r, c = queue.popleft()
        for dr, dc, expected_fdr, step_dist in UPSTREAM_CHECKS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < nrows and 0 <= nc < ncols):
                continue
            if dist[nr, nc] >= 0:
                continue  # already visited
            cell_fdr = int(fdr_arr[nr, nc])
            if fdr_nodata is not None and cell_fdr == int(fdr_nodata):
                continue
            if cell_fdr == expected_fdr:
                dist[nr, nc] = dist[r, c] + step_dist
                queue.append((nr, nc))

    # ------------------------------------------------------------------
    # Step 3: Headwater = cell with maximum distance from outlet
    # ------------------------------------------------------------------
    headwater_flat          = int(np.argmax(dist))
    headwater_r, headwater_c = divmod(headwater_flat, ncols)

    if dist[headwater_r, headwater_c] <= 0:
        return None

    # ------------------------------------------------------------------
    # Step 4: Traceback from headwater to outlet via D8 pointers
    # ------------------------------------------------------------------
    chain     = []
    r, c      = headwater_r, headwater_c
    max_steps = nrows * ncols  # safety cap against infinite loops

    for _ in range(max_steps):
        chain.append((r, c))
        if r == outlet_r and c == outlet_c:
            break
        fdr_val = int(fdr_arr[r, c])
        if fdr_nodata is not None and fdr_val == int(fdr_nodata):
            break
        offset = D8_OFFSETS.get(fdr_val)
        if offset is None:
            break
        nr, nc = r + offset[0], c + offset[1]
        if not (0 <= nr < nrows and 0 <= nc < ncols):
            break
        r, c = nr, nc

    if len(chain) < 2:
        return None

    # Reverse to mouth->headwater convention
    chain.reverse()
    return chain

def chain_to_linestring(chain: list, transform) -> LineString:
    """Convert a pixel chain [(r,c),...] to a map-coordinate LineString."""
    coords = [
        (transform.c + (c + 0.5) * transform.a,
         transform.f + (r + 0.5) * transform.e)
        for r, c in chain
    ]
    return LineString(coords)


def main():
    watersheds_dir = Path(WATERSHEDS_DIR)
    output_dir     = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    watersheds_shp = Path(WATERSHEDS_SHP)
    out_gpkg       = output_dir / OUTPUT_FILE

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not watersheds_shp.exists():
        logger.error(f"Watersheds shapefile not found: {watersheds_shp}")
        logger.error("Run delineate_watersheds.py first.")
        sys.exit(1)

    with fiona.open(str(watersheds_shp)) as shp:
        if ID_FIELD not in shp.schema["properties"]:
            logger.error(
                f"ID_FIELD '{ID_FIELD}' not found in shapefile. "
                f"Available: {list(shp.schema['properties'].keys())}"
            )
            sys.exit(1)
        wids = [int(feat["properties"][ID_FIELD]) for feat in shp]

    if not wids:
        logger.error("No watershed features found in shapefile.")
        sys.exit(1)

    # Match each ID to its clipped FDR and FAC rasters
    watersheds = []
    for wid in sorted(wids):
        fdr_path = watersheds_dir / f"watershed_{wid}_fdr.tif"
        fac_path = watersheds_dir / f"watershed_{wid}_fac.tif"
        if not fdr_path.exists():
            logger.warning(f"  FDR not found for watershed {wid}: {fdr_path.name} — skipping.")
            continue
        if not fac_path.exists():
            logger.warning(f"  FAC not found for watershed {wid}: {fac_path.name} — skipping.")
            continue
        watersheds.append((wid, fdr_path, fac_path))

    if not watersheds:
        logger.error(
            f"No watershed_{{wid}}_fdr.tif / _fac.tif pairs found in {watersheds_dir}. "
            "Run clip_watersheds.py first."
        )
        sys.exit(1)

    logger.info(f"Watersheds to process : {len(watersheds)}")
    logger.info(f"Output                : {out_gpkg}")
    logger.info(f"Min length            : {MIN_LENGTH_M} m")
    logger.info("-" * 60)

    # Read CRS from the first FDR raster (all share the same CRS)
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

    with fiona.open(
        str(out_gpkg), mode="w", driver="GPKG",
        schema=schema, crs=out_crs, layer="longest_branches",
    ) as dst:

        for wid, fdr_path, fac_path in watersheds:
            logger.info(f"Processing watershed {wid}...")
            t0 = time.time()

            try:
                with rasterio.open(str(fdr_path)) as src:
                    fdr_arr   = src.read(1)
                    fdr_nd    = src.nodata
                    transform = src.transform
                    cell_size = abs(src.res[0])

                with rasterio.open(str(fac_path)) as src:
                    fac_arr = src.read(1).astype(np.float64)
                    fac_nd  = src.nodata

                chain = longest_flowpath(fdr_arr, fac_arr, fdr_nd, fac_nd, cell_size)

                if chain is None:
                    logger.warning(f"  No valid flowpath found for watershed {wid}, skipping.")
                    skipped += 1
                    continue

                line = chain_to_linestring(chain, transform)

                if line.length < MIN_LENGTH_M:
                    logger.info(
                        f"  Watershed {wid}: {line.length:.1f} m "
                        f"< MIN_LENGTH_M ({MIN_LENGTH_M} m), skipping."
                    )
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

                logger.info(
                    f"  OK  watershed {wid}  |  "
                    f"{line.length / 1000:.2f} km  |  "
                    f"{len(line.coords)} vertices  |  "
                    f"{time.time() - t0:.1f}s"
                )
                succeeded += 1

            except Exception as e:
                logger.error(f"  ERROR processing watershed {wid}: {e}", exc_info=True)
                failed += 1

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info(f"  Output     : {out_gpkg}")
    logger.info(f"  Succeeded  : {succeeded}")
    logger.info(f"  Skipped    : {skipped}  (too short or no valid path)")
    logger.info(f"  Failed     : {failed}  (error)")
    logger.info(f"  Total time : {elapsed / 60:.1f} minutes")
    logger.info("")
    logger.info("Load longest_branches.gpkg in ArcGIS Pro / QGIS to verify.")
    logger.info("Each feature is a LineString ordered mouth->headwater.")


if __name__ == "__main__":
    main()