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
    - WhiteboxTools v2.4.0+ executable configured via config.WBT_EXE
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
# WBT SnapPourPoints expects map units, so this is multiplied by cell size
# internally. 50 cells * 2m = 100m snap radius.
SNAP_DISTANCE = config.SNAP_DISTANCE  # cells (e.g. 50 cells = 100 m at 2 m resolution)

# Set to True to log FDR diagnostics around the snapped pour point.
# Useful when WBT Watershed returns an empty result; disable for production runs.
DEBUG_POUR_POINTS = False

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
        import threading

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

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


def snap_pour_points_wbt(
    points_gdf: gpd.GeoDataFrame,
    fac_path: Path,
    snap_distance_cells: int,
    cell_size: float,
    temp_shp: Path,
    snapped_tif: Path,
    logger: logging.Logger,
) -> None:
    """
    Snap pour points to the nearest high-accumulation stream cell using WBT's
    own SnapPourPoints tool, then burn the result to a raster for Watershed.

    Using WBT's native vector snapping is more reliable than the numpy raster
    approach because it avoids the int32/float64 dtype mismatch that causes
    WBT Watershed to hang silently on some builds.

    snap_distance_cells * cell_size converts the cell-based snap distance to
    map units, which is what WBT SnapPourPoints expects.
    """
    snap_dist_m = snap_distance_cells * cell_size

    # Write the pour points to a temporary shapefile for WBT
    points_gdf[["POUR_ID", "geometry"]].to_file(str(temp_shp))
    logger.info(f"  Written {len(points_gdf)} pour point(s) to {temp_shp.name}")

    snapped_shp = temp_shp.with_name("pourpoints_snapped.shp")

    success = run_wbt("SnapPourPoints", {
        "pour_pts"  : str(temp_shp),
        "flow_accum": str(fac_path),
        "output"    : str(snapped_shp),
        "snap_dist" : snap_dist_m,
    }, logger)

    if not success or not snapped_shp.exists():
        logger.warning(
            "  WBT SnapPourPoints failed — falling back to unsnapped pour points. "
            "Watershed result may be inaccurate if points are not on stream cells."
        )
        snapped_shp = temp_shp

    # Burn snapped vector points to a raster aligned with the FAC grid
    snapped_gdf = gpd.read_file(str(snapped_shp))
    snapped_gdf["POUR_ID"] = range(1, len(snapped_gdf) + 1)
    _points_to_raster(snapped_gdf, "POUR_ID", fac_path, snapped_tif)
    logger.info(f"  Snapped pour point raster written: {snapped_tif.name}")


def _points_to_raster(
    points_gdf: gpd.GeoDataFrame,
    value_col: str,
    ref_raster_path: Path,
    out_path: Path,
) -> None:
    """
    Burn point values into a new raster whose grid exactly matches the
    reference raster. Using the WBT FAC as the reference guarantees cell
    alignment with every other WBT-produced raster in the pipeline.

    Written as float64 — WBT Watershed hangs silently on int32 pour point
    rasters in some v2.4.0 builds.
    """
    with rasterio.open(ref_raster_path) as src:
        meta      = src.meta.copy()
        transform = src.transform
        arr_shape = (src.height, src.width)

    meta.update(dtype=rasterio.float64, count=1, nodata=-9999.0)
    out_arr = np.full(arr_shape, -9999.0, dtype=np.float64)

    for _, row in points_gdf.iterrows():
        col_idx, row_idx = ~transform * (row.geometry.x, row.geometry.y)
        col_idx, row_idx = int(col_idx), int(row_idx)
        if 0 <= row_idx < arr_shape[0] and 0 <= col_idx < arr_shape[1]:
            out_arr[row_idx, col_idx] = float(row[value_col])

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(out_arr, 1)


def debug_pour_points(snapped_tif: Path, fdr_path: Path, logger: logging.Logger) -> None:
    """
    Log FDR diagnostics for every snapped pour point cell.
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
        logger.info(
            f"  Pour point ID={snapped_arr[r, c]} at "
            f"row={r}, col={c}, coords=({x:.1f}, {y:.1f})"
        )
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
        f"  First polygon raw area: {raw_area:,.0f} CRS units\u00b2 "
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
        f"@ {cell_w:.1f}m resolution = ~{min_area_km2:.2f} km\u00b2"
    )
    logger.info(
        f"Snap distance     : {SNAP_DISTANCE} cells "
        f"({SNAP_DISTANCE * cell_w:.0f} m)"
    )
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
    primary           = endpoints.sort_values("fac_value", ascending=False).iloc[[0]].copy()
    primary["POUR_ID"] = 1
    primary_accum     = int(primary["fac_value"].iloc[0])
    logger.info(f"  Primary outlet identified (Accumulation: {primary_accum:,} cells)")

    # ------------------------------------------------------------------
    # Step 2: Snap pour point with WBT SnapPourPoints, then rasterise
    # ------------------------------------------------------------------
    logger.info("Step 2: Snapping pour point with WBT SnapPourPoints...")

    temp_shp    = output_dir / "temp_pour_points.shp"
    snapped_tif = output_dir / "pourpoints_snapped.tif"

    snap_pour_points_wbt(
        primary, fac_path, SNAP_DISTANCE, cell_w,
        temp_shp, snapped_tif, logger
    )

    # Clean up all sidecar files produced by the shapefile write
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        for stem in ["temp_pour_points", "pourpoints_snapped"]:
            (output_dir / f"{stem}{ext}").unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Optional FDR diagnostics — set DEBUG_POUR_POINTS = True in CONFIG
    # ------------------------------------------------------------------
    if DEBUG_POUR_POINTS:
        logger.info("DEBUG: Inspecting FDR values at snapped pour point(s)...")
        debug_pour_points(snapped_tif, fdr_path, logger)

    # ------------------------------------------------------------------
    # Step 3 & 4: WBT watershed delineation -> vectorise -> area stats
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
            f"{row.geometry.area:,.0f} m\u00b2  ({row['area_km2']:.2f} km\u00b2)"
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
