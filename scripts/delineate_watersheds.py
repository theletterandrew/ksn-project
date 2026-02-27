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
    2. Run from the ArcGIS Pro Python environment (or any env with the deps):
       conda activate arcgispro-py3
       python delineate_watersheds.py

Requirements:
    - geopandas   (pip install geopandas)
    - rasterio    (pip install rasterio)
    - numpy       (pip install numpy)
    - WhiteboxTools executable on PATH or configured via config.WBT_EXE
    - Completed wbt_hydrology.py and stream_extraction_wbt.py first
    - streams_connected.gpkg produced by stream_extraction_wbt.py
"""

import logging
import sys
import time
import warnings
from pathlib import Path
import subprocess

import numpy as np
import rasterio
import rasterio.features
import geopandas as gpd
from shapely.geometry import shape as shapely_shape

# Calculate the path to the project root (one level up from scripts/)
root_dir = Path(__file__).resolve().parent.parent

# Add the root directory to sys.path so Python can find config.py
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

# =============================================================================
# CONFIG — Edit these before running
# =============================================================================

WBT_DIR     = config.DATA_SCRATCH_WBT
STREAMS_SHP = config.DATA_STREAMS / "streams_connected.gpkg"
OUTPUT_DIR  = config.DATA_SCRATCH_WATERSHEDS
WBT_EXE     = config.WBT_EXE

# Rasters produced by wbt_hydrology.py — all on the same grid, guaranteed
# to have matching cell alignment. This is what eliminates the "Invalid
# Pointer" / "no valid cells" errors caused by mixing WBT and arcpy surfaces.
FILLED_DEM = WBT_DIR / "dem_filled.tif"
FDR_FILE   = WBT_DIR / "flow_direction.tif"    # WBT D8 pointer raster
FAC_FILE   = WBT_DIR / "flow_accumulation.tif" # WBT flow accumulation

# Minimum drainage area threshold for watershed outlets.
# At 2m resolution:
#   10,000,000 cells  = ~40 km²   (large watersheds only)
#   25,000,000 cells  = ~100 km²  (major drainages)
#   50,000,000 cells  = ~200 km²  (very large basins)
MIN_DRAINAGE_AREA_CELLS = config.MIN_WATERSHED_AREA  # ~40 km² at 2m resolution

# Snap distance for pour points (cells).
# Pour points are snapped to the highest flow accumulation cell within
# this distance to ensure they land exactly on the stream.
SNAP_DISTANCE = config.SNAP_DISTANCE  # cells (e.g. 50 cells = 100 m at 2 m resolution)

# Set to True to print per-cell FDR diagnostics around the snapped pour point.
# Useful when WBT Watershed returns an empty result; disable for production runs.
DEBUG_POUR_POINTS = True

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
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def run_wbt(tool: str, args: dict, logger: logging.Logger, timeout: int = 600) -> bool:
    """
    Run a WhiteboxTools command via subprocess.
    Streams stdout/stderr in real time so progress is visible immediately.
    Kills the process and returns False if it exceeds `timeout` seconds.
    """
    cmd = [str(WBT_EXE), f"--run={tool}"]
    for key, val in args.items():
        cmd.append(f"--{key}={val}")
    logger.info(f"  Running WBT tool: {tool}")
    logger.info(f"  Command: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        import threading

        def stream_output(pipe, log_fn):
            for line in iter(pipe.readline, ""):
                line = line.rstrip()
                if line:
                    log_fn(f"    WBT: {line}")
            pipe.close()

        stdout_thread = threading.Thread(
            target=stream_output, args=(process.stdout, logger.info), daemon=True
        )
        stderr_thread = threading.Thread(
            target=stream_output, args=(process.stderr, logger.warning), daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()

        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            logger.error(
                f"  {tool} killed after {timeout}s timeout. "
                f"Check that pour points overlap the FDR raster and that "
                f"the WBT executable is not prompting for input."
            )
            return False

        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

        if process.returncode != 0:
            logger.error(f"  {tool} failed with return code {process.returncode}")
            return False

        return True

    except Exception as e:
        logger.error(f"  Failed to run {tool}: {e}")
        return False


def extract_stream_endpoints(streams_gpkg: Path) -> gpd.GeoDataFrame:
    """
    Return a GeoDataFrame of outlet points from the stream network GeoPackage.

    The stream network produced by stream_extraction_wbt.py contains Polygon
    geometries (stream corridor footprints). We derive outlet points by taking
    the centroid of each polygon, then keeping only the one with the highest
    flow accumulation value (sampled in the next step).
    """
    streams   = gpd.read_file(streams_gpkg, layer="streams")
    end_geoms = streams.geometry.centroid
    return gpd.GeoDataFrame(geometry=end_geoms, crs=streams.crs).reset_index(drop=True)


def sample_raster_at_points(
    raster_path: Path, points_gdf: gpd.GeoDataFrame, col_name: str
) -> gpd.GeoDataFrame:
    """Sample a raster at each point location and attach values as a new column."""
    with rasterio.open(raster_path) as src:
        coords = [(geom.x, geom.y) for geom in points_gdf.geometry]
        values = [v[0] for v in src.sample(coords)]
    gdf = points_gdf.copy()
    gdf[col_name] = values
    return gdf


def points_to_raster(
    points_gdf: gpd.GeoDataFrame,
    value_col: str,
    ref_raster_path: Path,
    out_path: Path,
) -> None:
    """
    Burn point values into a new raster whose grid exactly matches the
    reference raster. Using the WBT FAC as the reference guarantees cell
    alignment with every other WBT-produced raster in the pipeline.
    """
    with rasterio.open(ref_raster_path) as src:
        meta      = src.meta.copy()
        transform = src.transform
        arr_shape = (src.height, src.width)

    meta.update(dtype=rasterio.int32, count=1, nodata=-9999)
    out_arr = np.full(arr_shape, -9999, dtype=np.int32)

    for _, row in points_gdf.iterrows():
        col_idx, row_idx = ~transform * (row.geometry.x, row.geometry.y)
        col_idx, row_idx = int(col_idx), int(row_idx)
        if 0 <= row_idx < arr_shape[0] and 0 <= col_idx < arr_shape[1]:
            out_arr[row_idx, col_idx] = int(row[value_col])

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(out_arr, 1)


def snap_pour_points(
    pour_raster_path: Path,
    fac_path: Path,
    snap_distance: int,
    out_path: Path,
    logger: logging.Logger,
) -> None:
    """
    For each labelled cell in the pour raster, locate the cell with the
    highest flow accumulation within snap_distance cells and move the pour
    point there. Pure numpy — no arcpy required, no grid mismatch possible
    because both inputs come from the same WBT pipeline.
    """
    with rasterio.open(pour_raster_path) as src:
        pour_arr = src.read(1).astype(np.int32)
        nodata   = int(src.nodata) if src.nodata is not None else -9999
        meta     = src.meta.copy()

    # Use float32 for FAC — halves memory vs float64 with no precision impact
    # for the argmax comparison we do here.
    with rasterio.open(fac_path) as src:
        fac_arr = src.read(1).astype(np.float32)
        fac_nd  = src.nodata

    if fac_nd is not None:
        fac_arr[fac_arr == fac_nd] = -1.0

    snapped   = np.full_like(pour_arr, nodata)
    pour_rows, pour_cols = np.where(pour_arr != nodata)

    if len(pour_rows) == 0:
        logger.error(
            "Pour point raster contains no valid cells. "
            "Verify that stream endpoints fall within the FAC raster extent "
            "and that both datasets share the same CRS."
        )
        sys.exit(1)

    for r, c in zip(pour_rows, pour_cols):
        val = pour_arr[r, c]
        r0  = max(0, r - snap_distance)
        r1  = min(fac_arr.shape[0], r + snap_distance + 1)
        c0  = max(0, c - snap_distance)
        c1  = min(fac_arr.shape[1], c + snap_distance + 1)
        window = fac_arr[r0:r1, c0:c1]
        best   = np.unravel_index(np.argmax(window), window.shape)
        snapped[r0 + best[0], c0 + best[1]] = val

    logger.info(f"  Snapped {len(pour_rows)} pour point(s) within {snap_distance} cells")

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(snapped, 1)


def debug_pour_points(snapped_tif: Path, fdr_path: Path, logger: logging.Logger) -> None:
    """
    Print FDR diagnostics for every snapped pour point cell.
    Only called when DEBUG_POUR_POINTS = True.
    """
    with rasterio.open(snapped_tif) as src:
        snapped_arr = src.read(1)
        nodata      = src.nodata
        transform   = src.transform
        rows, cols  = np.where(snapped_arr != nodata)

    with rasterio.open(fdr_path) as src:
        fdr_arr    = src.read(1)
        fdr_nodata = src.nodata

    for r, c in zip(rows, cols):
        x, y    = transform * (c, r)
        fdr_val = fdr_arr[r, c]
        logger.info(f"  Pour point ID={snapped_arr[r, c]} at row={r}, col={c}, coords=({x:.1f}, {y:.1f})")
        if fdr_val == 0:
            logger.warning(f"    FDR={fdr_val} — pour point is on a zero/flat cell!")
        elif fdr_nodata is not None and fdr_val == fdr_nodata:
            logger.warning(f"    FDR={fdr_val} — pour point is on a nodata cell!")
        else:
            logger.info(f"    FDR={fdr_val} — valid")
        window = fdr_arr[max(0, r - 3):r + 4, max(0, c - 3):c + 4]
        logger.info(f"    FDR values in 7x7 neighbourhood:\n{window}")


def delineate_and_vectorise(
    fdr_path: Path,
    pour_raster_path: Path,
    watersheds_tif: Path,
    watersheds_shp: Path,
    logger: logging.Logger,
) -> gpd.GeoDataFrame:
    """
    Run WBT watershed delineation, vectorise the result with rasterio, and
    return a GeoDataFrame with area_km2 attached.
    """
    logger.info("  Running WBT Watershed tool...")
    success = run_wbt("Watershed", {
        "d8_pntr"  : str(fdr_path),
        "pour_pts" : str(pour_raster_path),
        "output"   : str(watersheds_tif),
    }, logger)
    if not success:
        logger.error("WBT Watershed tool failed.")
        sys.exit(1)

    if not watersheds_tif.exists():
        logger.error(f"WBT Watershed produced no output at {watersheds_tif}.")
        sys.exit(1)

    logger.info("  Vectorising watershed raster...")
    with rasterio.open(watersheds_tif) as src:
        arr       = src.read(1)
        nodata    = src.nodata
        crs       = src.crs
        transform = src.transform

    valid_mask = (
        (arr != nodata).astype(np.uint8)
        if nodata is not None
        else np.ones(arr.shape, dtype=np.uint8)
    )

    # Use a generator to avoid loading all shapes into memory at once
    shapes_gen = rasterio.features.shapes(
        arr.astype(np.int32), mask=valid_mask, transform=transform
    )

    # Silence the rasterio/GDAL 3.11 'Memory' driver deprecation warning —
    # it is harmless and will be fixed in a future rasterio release.
    warnings.filterwarnings(
        "ignore",
        message=".*Memory.*driver is deprecated.*",
        category=RuntimeWarning,
    )

    geoms, gridcodes = [], []
    for geom_dict, val in shapes_gen:
        geoms.append(shapely_shape(geom_dict))
        gridcodes.append(int(val))

    if not geoms:
        logger.error(
            "Vectorisation produced no polygons — the watershed raster appears empty. "
            "Check that the pour point raster overlaps the flow direction raster."
        )
        sys.exit(1)

    gdf = gpd.GeoDataFrame({"gridcode": gridcodes}, geometry=geoms, crs=crs)

    # Dissolve so each watershed ID becomes a single polygon
    gdf = gdf.dissolve(by="gridcode").reset_index()

    # Sanity-check the first polygon's area
    raw_area = gdf.geometry.area.iloc[0]
    logger.info(
        f"  First polygon raw area: {raw_area:,.0f} CRS units² "
        f"(CRS: {crs.to_string() if crs else 'unknown'})"
    )

    if raw_area < 1:
        logger.error(
            "  Polygon area is effectively zero. The watershed raster likely "
            "contains only a single cell or the geometry was not reprojected. "
            "Verify that the FDR and pour point rasters share the same CRS and "
            "that the outlet cell is not on the raster boundary."
        )
        sys.exit(1)

    gdf["area_km2"] = gdf.geometry.area / 1e6
    gdf.to_file(watersheds_shp)
    return gdf


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    fdr_path    = Path(FDR_FILE)
    fac_path    = Path(FAC_FILE)
    filled_dem  = Path(FILLED_DEM)
    streams_shp = Path(STREAMS_SHP)

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    missing = [
        f"  {lbl}: {p}"
        for lbl, p in [
            ("Flow direction (FDR)", fdr_path),
            ("Flow accumulation (FAC)", fac_path),
            ("Filled DEM", filled_dem),
            ("Stream network", streams_shp),
        ]
        if not p.exists()
    ]
    if missing:
        for m in missing:
            logger.error(m)
        logger.error("Run wbt_hydrology.py and stream_extraction_wbt.py first.")
        sys.exit(1)

    # Read actual cell size from the FAC raster rather than hardcoding 2m
    with rasterio.open(fac_path) as src:
        cell_w, cell_h = abs(src.res[0]), abs(src.res[1])
        cell_area_km2  = (cell_w * cell_h) / 1e6

    min_area_km2 = MIN_DRAINAGE_AREA_CELLS * cell_area_km2

    logger.info(
        f"Min drainage area : {MIN_DRAINAGE_AREA_CELLS:,} cells "
        f"@ {cell_w:.1f}m resolution = ~{min_area_km2:.2f} km²"
    )
    logger.info(f"Snap distance     : {SNAP_DISTANCE} cells ({SNAP_DISTANCE * cell_w:.0f} m)")
    logger.info("-" * 60)

    start_time = time.time()

    # ------------------------------------------------------------------
    # Step 1: Extract stream endpoints and sample FAC values
    # ------------------------------------------------------------------
    logger.info("Step 1: Identifying the primary stream outlet...")

    endpoints = extract_stream_endpoints(streams_shp)
    endpoints = sample_raster_at_points(fac_path, endpoints, "fac_value")

    # Drop any points that landed outside the raster (returned nodata)
    with rasterio.open(fac_path) as src:
        fac_nodata = src.nodata
    if fac_nodata is not None:
        endpoints = endpoints[endpoints["fac_value"] != fac_nodata]

    if endpoints.empty:
        logger.error(
            "No stream endpoints overlapped the FAC raster. "
            "Confirm that streams_connected.gpkg and flow_accumulation.tif "
            "share the same CRS and cover the same area."
        )
        sys.exit(1)

    # Keep only the single outlet with the highest flow accumulation
    primary        = endpoints.sort_values("fac_value", ascending=False).iloc[[0]].copy()
    primary["POUR_ID"] = 1
    primary_accum  = int(primary["fac_value"].iloc[0])
    logger.info(f"  Primary outlet identified (Accumulation: {primary_accum:,} cells)")

    # ------------------------------------------------------------------
    # Step 2: Burn pour point to raster aligned with WBT FAC, then snap
    # ------------------------------------------------------------------
    logger.info("Step 2: Rasterising and snapping pour point...")

    temp_pour_raster = output_dir / "temp_pour_points.tif"
    snapped_tif      = output_dir / "pourpoints_snapped.tif"

    points_to_raster(primary, "POUR_ID", fac_path, temp_pour_raster)
    snap_pour_points(temp_pour_raster, fac_path, SNAP_DISTANCE, snapped_tif, logger)

    temp_pour_raster.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Optional FDR diagnostics — set DEBUG_POUR_POINTS = True in CONFIG
    # ------------------------------------------------------------------
    if DEBUG_POUR_POINTS:
        logger.info("DEBUG: Inspecting FDR values at snapped pour point(s)...")
        debug_pour_points(snapped_tif, fdr_path, logger)

    # ------------------------------------------------------------------
    # Step 3 & 4: WBT watershed delineation → vectorise → area stats
    # ------------------------------------------------------------------
    logger.info("Step 3: Delineating watershed with WhiteboxTools...")

    watersheds_tif = output_dir / "watersheds.tif"
    watersheds_shp = output_dir / "watersheds.shp"

    gdf = delineate_and_vectorise(
        fdr_path, snapped_tif, watersheds_tif, watersheds_shp, logger
    )

    watershed_count = len(gdf)
    logger.info(f"  Created {watershed_count} watershed polygon(s)")

    # ------------------------------------------------------------------
    # Step 5: Report area statistics
    # ------------------------------------------------------------------
    logger.info("Step 5: Watershed area statistics:")
    for _, row in gdf.iterrows():
        logger.info(
            f"  Watershed {int(row['gridcode'])}: "
            f"{row.geometry.area:,.0f} m²  ({row['area_km2']:.2f} km²)"
        )

    # ------------------------------------------------------------------
    # Clean up intermediate rasters
    # ------------------------------------------------------------------
    logger.info("Cleaning up intermediate files...")
    for tmp in [snapped_tif, watersheds_tif]:
        tmp.unlink(missing_ok=True)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info(f"  Output          : {watersheds_shp}")
    logger.info(f"  Watershed count : {watershed_count}")
    logger.info(f"  Total time      : {elapsed / 60:.1f} minutes")
    logger.info("")
    logger.info("Load watersheds.shp in ArcGIS Pro to visualize.")
    logger.info("Use these polygons to clip DEMs for individual ksn analysis.")


if __name__ == "__main__":
    main()
