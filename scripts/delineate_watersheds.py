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


def extract_stream_outlets(streams_gpkg: Path, fac_path: Path, sample_spacing_m: float = 10.0) -> gpd.GeoDataFrame:
    """
    Return a GeoDataFrame with one outlet point per stream segment — the point
    along each line that has the highest FAC value.

    Strategy: interpolate candidate points along each line at `sample_spacing_m`
    intervals (default every 10 m, i.e. every 5 cells at 2 m resolution), then
    sample the FAC raster at all of them in a single vectorised pass and keep
    the best point per segment. This is robust to:
      - lines stored in source->mouth or mouth->source order
      - endpoints that fall on headwater cells well away from the main trunk
      - stream networks whose geometry does not extend all the way to the outlet

    For Polygon geometries (older pipeline) falls back to centroid.
    """
    from shapely.geometry import MultiLineString, Point

    streams   = gpd.read_file(streams_gpkg, layer="streams")
    geom_type = streams.geometry.geom_type.iloc[0] if not streams.empty else "Unknown"

    if "Line" not in geom_type:
        end_geoms = streams.geometry.centroid
        return gpd.GeoDataFrame(geometry=end_geoms, crs=streams.crs).reset_index(drop=True)

    # ---------------------------------------------------------------
    # Build a flat list of candidate points with a back-reference to
    # which stream segment they came from.
    # ---------------------------------------------------------------
    seg_ids   = []   # which segment index each candidate belongs to
    cand_pts  = []   # shapely Points

    for seg_idx, geom in enumerate(streams.geometry):
        length = geom.length
        if length == 0:
            # Degenerate line — just use the single coordinate
            coords = list(geom.coords) if not isinstance(geom, MultiLineString)                      else list(geom.geoms[0].coords)
            cand_pts.append(Point(coords[0]))
            seg_ids.append(seg_idx)
            continue

        # Interpolate at regular intervals plus both endpoints
        n_steps   = max(1, int(length / sample_spacing_m))
        distances = [i * length / n_steps for i in range(n_steps + 1)]
        for d in distances:
            pt = geom.interpolate(d)
            cand_pts.append(pt)
            seg_ids.append(seg_idx)

    # ---------------------------------------------------------------
    # Sample FAC at all candidate points in a single rasterio call
    # ---------------------------------------------------------------
    with rasterio.open(fac_path) as src:
        nodata   = src.nodata
        fac_vals = [v[0] for v in src.sample([(p.x, p.y) for p in cand_pts])]

    # ---------------------------------------------------------------
    # For each segment keep the candidate with the highest valid FAC
    # ---------------------------------------------------------------
    n_segs      = len(streams)
    best_pt     = [None]    * n_segs
    best_fac    = [-1.0]    * n_segs

    for pt, seg_idx, fval in zip(cand_pts, seg_ids, fac_vals):
        if nodata is not None and fval == nodata:
            continue
        if fval > best_fac[seg_idx]:
            best_fac[seg_idx]  = fval
            best_pt[seg_idx]   = pt

    # Fall back to midpoint for any segment that got no valid sample
    chosen = []
    for seg_idx, (pt, geom) in enumerate(zip(best_pt, streams.geometry)):
        if pt is None:
            pt = geom.interpolate(0.5, normalized=True)
        chosen.append(pt)

    return gpd.GeoDataFrame(geometry=chosen, crs=streams.crs).reset_index(drop=True)


def find_outlets_from_fac(
    fac_path: Path,
    fdr_path: Path,
    min_accum_cells: int,
    min_sep_m: float,
    logger: logging.Logger,
) -> gpd.GeoDataFrame:
    """
    Derive pour points directly from the FAC and FDR rasters, bypassing the
    stream vector layer entirely.

    A cell is treated as an outlet if it meets ALL of these criteria:
      1. FAC >= min_accum_cells  (large enough drainage area)
      2. It is a local FAC maximum within min_sep_m radius  (avoids duplicates
         on the same trunk — keeps only the furthest-downstream cell per stream)
      3. Its D8 flow direction points OUT of the raster, OR it drains into a
         nodata cell  (i.e. it is a true outlet at the study-area boundary)

    If criterion 3 yields no outlets (closed basin / internal drainage), the
    function falls back to the single cell with the highest FAC value so the
    pipeline always produces at least one watershed.
    """
    from rasterio.transform import xy as rio_xy

    with rasterio.open(fac_path) as src:
        fac_arr   = src.read(1).astype(np.float64)
        fac_nd    = src.nodata
        transform = src.transform
        crs       = src.crs
        nrows, ncols = src.shape

    with rasterio.open(fdr_path) as src:
        fdr_arr = src.read(1)
        fdr_nd  = src.nodata

    if fac_nd is not None:
        fac_arr[fac_arr == fac_nd] = 0

    # WBT D8 pointer values and their (row, col) offsets
    D8_OFFSETS = {
        1:   ( 0,  1),   # E
        2:   ( 1,  1),   # SE
        4:   ( 1,  0),   # S
        8:   ( 1, -1),   # SW
        16:  ( 0, -1),   # W
        32:  (-1, -1),   # NW
        64:  (-1,  0),   # N
        128: (-1,  1),   # NE
    }

    # ------------------------------------------------------------------
    # Criterion 1: cells above the accumulation threshold
    # ------------------------------------------------------------------
    rows_above, cols_above = np.where(fac_arr >= min_accum_cells)
    if len(rows_above) == 0:
        logger.error(
            f"No FAC cells >= {min_accum_cells:,}. "
            "Check MIN_WATERSHED_AREA in config.py."
        )
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    logger.info(f"  {len(rows_above):,} FAC cells >= {min_accum_cells:,} cells")

    # ------------------------------------------------------------------
    # Criterion 2: local FAC maxima within min_sep_m radius (fast numpy)
    # Keep a cell only if no neighbour within the radius has a higher FAC.
    # We approximate the radius as a square window for speed.
    # ------------------------------------------------------------------
    with rasterio.open(fac_path) as src:
        cell_size = abs(src.res[0])
    half_win = max(1, int(min_sep_m / cell_size))

    local_max_mask = np.zeros(fac_arr.shape, dtype=bool)
    for r, c in zip(rows_above, cols_above):
        r0, r1 = max(0, r - half_win), min(nrows, r + half_win + 1)
        c0, c1 = max(0, c - half_win), min(ncols, c + half_win + 1)
        if fac_arr[r, c] == fac_arr[r0:r1, c0:c1].max():
            local_max_mask[r, c] = True

    rows_lm, cols_lm = np.where(local_max_mask)
    logger.info(f"  {len(rows_lm):,} local FAC maxima after deduplication")

    # Criterion 3 (restored): cell's D8 pointer exits the raster
    # OR drains into a nodata cell — these are the true basin outlets
    boundary_outlets = []
    for r, c in zip(rows_lm, cols_lm):
        fdr_val = int(fdr_arr[r, c])
        if fdr_nodata is not None and fdr_val == int(fdr_nodata):
            continue
        offset = D8_OFFSETS.get(fdr_val)
        if offset is None:
            continue
        nr, nc = r + offset[0], c + offset[1]
        # Exits raster boundary
        exits_boundary = not (0 <= nr < nrows and 0 <= nc < ncols)
        # Drains into nodata
        drains_to_nodata = (
            (0 <= nr < nrows and 0 <= nc < ncols)
            and fdr_nd is not None
            and int(fdr_arr[nr, nc]) == int(fdr_nd)
        )
        if exits_boundary or drains_to_nodata:
            boundary_outlets.append((r, c))

    if boundary_outlets:
        outlet_rows = [r for r, c in boundary_outlets]
        outlet_cols = [c for r, c in boundary_outlets]
        logger.info(f"  {len(outlet_rows)} boundary outlet(s) identified")
    else:
        # Closed basin fallback — keep local maxima
        logger.warning("  No boundary outlets found — using local FAC maxima (closed basin)")
        outlet_rows, outlet_cols = rows_lm.tolist(), cols_lm.tolist()

    # ------------------------------------------------------------------
    # Outlets: the local FAC maxima are the outlets directly. This works
    # for both clipped tiles and closed basins.
    # ------------------------------------------------------------------

    # Convert row/col indices to map coordinates
    xs, ys = rio_xy(transform, outlet_rows, outlet_cols)
    points = [gpd.points_from_xy([x], [y])[0] for x, y in zip(xs, ys)]

    gdf = gpd.GeoDataFrame(
        {"fac_value": [float(fac_arr[r, c]) for r, c in zip(outlet_rows, outlet_cols)]},
        geometry=points,
        crs=crs,
    )
    logger.info(
        f"  Outlet FAC range: "
        f"min={gdf['fac_value'].min():,.0f}  "
        f"max={gdf['fac_value'].max():,.0f}"
    )
    return gdf

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

    # Ensure pour points share the FAC raster CRS before writing to disk —
    # WBT will silently sample wrong cells if the CRS differs.
    with rasterio.open(fac_path) as src:
        fac_crs = src.crs
    if points_gdf.crs and fac_crs and points_gdf.crs != fac_crs:
        logger.warning(
            f"  Reprojecting pour points to FAC CRS before SnapPourPoints "
            f"({points_gdf.crs.to_epsg()} -> {fac_crs.to_epsg()})"
        )
        points_gdf = points_gdf.to_crs(fac_crs)

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
    if snapped_gdf.empty:
        logger.warning(
            "  SnapPourPoints produced an empty shapefile — "
            "falling back to original (unsnapped) pour points."
        )
        snapped_gdf = gpd.read_file(str(temp_shp))

    snapped_gdf["POUR_ID"] = range(1, len(snapped_gdf) + 1)
    logger.info(f"  Burning {len(snapped_gdf)} snapped pour point(s) to raster...")
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

    Written as int16 — WBT Watershed requires an integer pour point raster.
    float64 causes a silent hang on most WBT builds; int32 causes it on some.
    int16 works reliably across all known versions and is what the WBT docs
    specify for pour point inputs.
    """
    with rasterio.open(ref_raster_path) as src:
        meta      = src.meta.copy()
        transform = src.transform
        arr_shape = (src.height, src.width)

    NODATA_VAL = -9999
    meta.update(dtype=rasterio.int16, count=1, nodata=NODATA_VAL)
    out_arr = np.full(arr_shape, NODATA_VAL, dtype=np.int16)

    burned = 0
    for _, row in points_gdf.iterrows():
        col_idx, row_idx = ~transform * (row.geometry.x, row.geometry.y)
        col_idx, row_idx = int(col_idx), int(row_idx)
        if 0 <= row_idx < arr_shape[0] and 0 <= col_idx < arr_shape[1]:
            out_arr[row_idx, col_idx] = int(row[value_col])
            burned += 1

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(out_arr, 1)

    if burned == 0:
        raise RuntimeError(
            f"_points_to_raster: no points fell within the raster extent. "
            f"Check that pour point coordinates match the reference raster CRS."
        )


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



def _d8_watershed_numpy(
    fdr_arr: np.ndarray,
    fdr_nodata,
    outlet_rows: list,
    outlet_cols: list,
) -> np.ndarray:
    """
    Pure-numpy D8 watershed delineation via BFS (flood-fill upstream).

    For each outlet, walks backwards up the D8 pointer grid collecting every
    cell that drains to that outlet. Returns an int32 label array where
    0 = unassigned and 1..N = watershed ID.

    WBT D8 pointer convention (powers of 2):
        1=E  2=SE  4=S  8=SW  16=W  32=NW  64=N  128=NE

    To find cells draining INTO (r, c) we check all 8 neighbours: a neighbour
    at offset (dr, dc) drains into (r, c) when its FDR value equals the
    direction pointing back toward (r, c), i.e. D8[(-dr, -dc)].
    """
    from collections import deque

    nrows, ncols = fdr_arr.shape

    D8 = {
        ( 0,  1): 1,
        ( 1,  1): 2,
        ( 1,  0): 4,
        ( 1, -1): 8,
        ( 0, -1): 16,
        (-1, -1): 32,
        (-1,  0): 64,
        (-1,  1): 128,
    }
    # (dr, dc, expected_fdr_in_neighbour) — neighbour drains into current cell
    UPSTREAM = [
        (dr, dc, D8[(-dr, -dc)])
        for (dr, dc) in D8
        if (-dr, -dc) in D8
    ]

    labels = np.zeros(fdr_arr.shape, dtype=np.int32)

    for ws_id, (out_r, out_c) in enumerate(zip(outlet_rows, outlet_cols), start=1):
        queue = deque()
        queue.append((out_r, out_c))
        labels[out_r, out_c] = ws_id

        while queue:
            r, c = queue.popleft()
            for dr, dc, expected_fdr in UPSTREAM:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < nrows and 0 <= nc < ncols):
                    continue
                if labels[nr, nc] != 0:
                    continue
                cell_fdr = int(fdr_arr[nr, nc])
                if fdr_nodata is not None and cell_fdr == int(fdr_nodata):
                    continue
                if cell_fdr == expected_fdr:
                    labels[nr, nc] = ws_id
                    queue.append((nr, nc))

    return labels


def delineate_and_vectorise(
    fdr_path: Path,
    pour_shp_path: Path,
    watersheds_tif: Path,
    watersheds_shp: Path,
    logger: logging.Logger,
) -> gpd.GeoDataFrame:
    """
    Delineate watersheds using a pure-numpy D8 BFS — no WBT Watershed call.
    Vectorise the result with rasterio and return a GeoDataFrame.
    """
    logger.info("  Loading FDR raster...")
    with rasterio.open(fdr_path) as src:
        fdr_arr   = src.read(1)
        fdr_nd    = src.nodata
        crs       = src.crs
        transform = src.transform
        fdr_meta  = src.meta.copy()

    pour_gdf = gpd.read_file(str(pour_shp_path))
    outlet_rows, outlet_cols = [], []
    for geom in pour_gdf.geometry:
        col, row = ~transform * (geom.x, geom.y)
        outlet_rows.append(int(row))
        outlet_cols.append(int(col))

    logger.info(
        f"  Running numpy D8 BFS watershed delineation "
        f"({len(outlet_rows)} outlet(s), "
        f"{fdr_arr.shape[1]}x{fdr_arr.shape[0]} grid)..."
    )
    labels = _d8_watershed_numpy(fdr_arr, fdr_nd, outlet_rows, outlet_cols)

    labeled_cells = int((labels > 0).sum())
    logger.info(f"  BFS complete — {labeled_cells:,} cells assigned to watersheds")

    if labeled_cells == 0:
        logger.error(
            "BFS returned no watershed cells. Verify the outlet cell has a "
            "valid non-zero FDR value and lies within the FDR raster extent."
        )
        sys.exit(1)

    fdr_meta.update(dtype=rasterio.int32, nodata=0)
    with rasterio.open(watersheds_tif, "w", **fdr_meta) as dst:
        dst.write(labels, 1)

    logger.info("  Vectorising watershed raster...")
    valid_mask = (labels != 0).astype(np.uint8)

    warnings.filterwarnings(
        "ignore",
        message=".*Memory.*driver is deprecated.*",
        category=RuntimeWarning,
    )

    shapes_gen = rasterio.features.shapes(
        labels.astype(np.int32), mask=valid_mask, transform=transform
    )

    geoms, gridcodes = [], []
    for geom_dict, val in shapes_gen:
        geoms.append(shapely_shape(geom_dict))
        gridcodes.append(int(val))

    if not geoms:
        logger.error("Vectorisation produced no polygons.")
        sys.exit(1)

    gdf = gpd.GeoDataFrame({"gridcode": gridcodes}, geometry=geoms, crs=crs)
    gdf = gdf.dissolve(by="gridcode").reset_index()

    raw_area = gdf.geometry.area.iloc[0]
    logger.info(
        f"  First polygon raw area: {raw_area:,.0f} CRS units\u00b2 "
        f"(CRS: {crs.to_string() if crs else 'unknown'})"
    )

    if raw_area < 1:
        logger.error(
            "Polygon area is effectively zero — check FDR raster and outlet location."
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

    # ------------------------------------------------------------------
    # FAC raster sanity check — runs before any endpoint logic so a bad
    # raster is caught immediately with actionable numbers.
    # ------------------------------------------------------------------
    logger.info("Checking FAC raster integrity...")
    with rasterio.open(fac_path) as src:
        _fac_arr  = src.read(1).astype(np.float64)
        _fac_nd   = src.nodata
        _fac_res  = src.res
        _fac_shape = src.shape
    if _fac_nd is not None:
        _fac_valid = _fac_arr[_fac_arr != _fac_nd]
    else:
        _fac_valid = _fac_arr.flatten()
    _fac_valid = _fac_valid[_fac_valid > 0]
    if _fac_valid.size == 0:
        logger.error(
            "FAC raster contains no positive values — "
            "wbt_hydrology.py may not have run successfully. "
            f"Raster: {fac_path}"
        )
        sys.exit(1)
    logger.info(
        f"  FAC raster : {_fac_shape[1]}x{_fac_shape[0]} px @ {_fac_res[0]:.1f}m  "
        f"valid cells={_fac_valid.size:,}  "
        f"max={_fac_valid.max():,.0f}  "
        f"p99={np.percentile(_fac_valid, 99):,.0f}  "
        f"p50={np.percentile(_fac_valid, 50):,.0f}"
    )
    _expected_max = (_fac_shape[0] * _fac_shape[1]) * 0.5   # rough lower bound
    if _fac_valid.max() < _expected_max * 0.001:
        logger.warning(
            f"  FAC maximum ({_fac_valid.max():,.0f}) looks suspiciously low for a "
            f"{_fac_shape[1]}x{_fac_shape[0]} raster. The flow accumulation may not "
            f"have been computed correctly — check wbt_hydrology.py outputs."
        )
    del _fac_arr, _fac_valid  # free memory before processing

    start_time = time.time()

    # ------------------------------------------------------------------
    # Step 1: Extract stream endpoints and sample FAC values
    # ------------------------------------------------------------------
    logger.info("Step 1: Identifying all qualifying stream outlets from FAC raster...")

    # Derive outlets directly from the FAC + FDR rasters rather than from the
    # stream vector layer. The stream GeoPackage may only cover headwater
    # segments and never intersect the high-accumulation trunk cells, so
    # vector-based outlet detection is unreliable for full-DEM tiling.
    min_sep_m = SNAP_DISTANCE * cell_w * 3  # minimum distance between outlets

    primary = find_outlets_from_fac(
        fac_path, fdr_path, MIN_DRAINAGE_AREA_CELLS, min_sep_m, logger
    )

    if primary.empty:
        logger.error(
            f"find_outlets_from_fac returned no outlets for threshold "
            f"{MIN_DRAINAGE_AREA_CELLS:,} cells. Lower MIN_WATERSHED_AREA in config.py."
        )
        sys.exit(1)

    primary = primary.sort_values("fac_value", ascending=False).reset_index(drop=True)
    primary["POUR_ID"] = range(1, len(primary) + 1)
    logger.info(
        f"  {len(primary)} outlet(s) selected "
        f"(FAC >= {MIN_DRAINAGE_AREA_CELLS:,} cells, "
        f"min separation {min_sep_m:.0f} m)"
    )

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
    # Validate snapped raster before calling WBT Watershed.
    # WBT hangs silently when the pour point raster has no cells that
    # overlap valid FDR data — catch this early with a clear error.
    # ------------------------------------------------------------------
    logger.info("  Validating snapped pour point raster...")
    with rasterio.open(snapped_tif) as _pp_src:
        _pp_arr    = _pp_src.read(1)
        _pp_nd     = _pp_src.nodata
        _pp_trans  = _pp_src.transform
        _pp_bounds = _pp_src.bounds

    _valid_pp = (_pp_arr != _pp_nd) if _pp_nd is not None else (_pp_arr > 0)
    _n_pp_cells = int(_valid_pp.sum())
    logger.info(f"  Pour point raster: {_n_pp_cells} valid cell(s)")

    if _n_pp_cells == 0:
        logger.error(
            "Snapped pour point raster is empty — no cells were burned. "
            "The pour point may lie outside the FAC raster extent. "
            "Check that find_outlets_from_fac returned coordinates inside "
            "the raster bounds."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Validate FDR at pour point and relocate if FDR == 0.
    # WBT uses 0 for flat/unresolved cells (often the raster edge row).
    # Walk one cell inward along the steepest FAC gradient until we find
    # a cell with a valid non-zero FDR value, then re-burn the raster.
    # ------------------------------------------------------------------
    with rasterio.open(fdr_path) as _fdr_src:
        _fdr_bounds  = _fdr_src.bounds
        _fdr_arr     = _fdr_src.read(1)
        _fdr_nd      = _fdr_src.nodata
        _fdr_trans   = _fdr_src.transform
        _fdr_nrows, _fdr_ncols = _fdr_src.shape

    with rasterio.open(fac_path) as _fac_src2:
        _fac_arr2 = _fac_src2.read(1).astype(np.float64)
        _fac_nd2  = _fac_src2.nodata
        if _fac_nd2 is not None:
            _fac_arr2[_fac_arr2 == _fac_nd2] = 0

    # Valid FDR: non-nodata AND non-zero (0 = flat/sink in WBT)
    def _fdr_valid(val):
        if _fdr_nd is not None and val == _fdr_nd:
            return False
        return int(val) != 0

    _pp_rows, _pp_cols = np.where(_valid_pp)
    _fixed_points = []   # (row, col, x, y, fdr_val) for each pour point

    for _r, _c in zip(_pp_rows, _pp_cols):
        _x, _y   = _pp_trans * (_c + 0.5, _r + 0.5)
        # Find this cell in the FDR grid
        _fc, _fr = ~_fdr_trans * (_x, _y)
        _fr, _fc = int(_fr), int(_fc)

        if not (0 <= _fr < _fdr_nrows and 0 <= _fc < _fdr_ncols):
            logger.error(f"  Pour point ({_x:.1f}, {_y:.1f}) is outside FDR extent.")
            sys.exit(1)

        _fdr_val = int(_fdr_arr[_fr, _fc])
        if _fdr_valid(_fdr_val):
            logger.info(
                f"  Pour point at ({_x:.1f}, {_y:.1f}) — FDR value: {_fdr_val} ✓"
            )
            _fixed_points.append((_fr, _fc, _x, _y, _fdr_val))
            continue

        # FDR is 0 — search an expanding neighbourhood (up to 20 cells)
        # for the neighbour with the highest FAC and a valid FDR.
        logger.warning(
            f"  Pour point at ({_x:.1f}, {_y:.1f}) has FDR=0 (flat/edge cell). "
            f"Searching for nearest valid FDR cell..."
        )
        _best_r, _best_c, _best_fdr, _best_fac = None, None, None, -1
        for _radius in range(1, 21):
            for _dr in range(-_radius, _radius + 1):
                for _dc in range(-_radius, _radius + 1):
                    if abs(_dr) != _radius and abs(_dc) != _radius:
                        continue   # only check the perimeter of the square
                    _nr, _nc = _fr + _dr, _fc + _dc
                    if not (0 <= _nr < _fdr_nrows and 0 <= _nc < _fdr_ncols):
                        continue
                    _nfdr = int(_fdr_arr[_nr, _nc])
                    _nfac = float(_fac_arr2[_nr, _nc])
                    if _fdr_valid(_nfdr) and _nfac > _best_fac:
                        _best_r, _best_c = _nr, _nc
                        _best_fdr, _best_fac = _nfdr, _nfac
            if _best_r is not None:
                break   # found a valid cell at this radius

        if _best_r is None:
            logger.error(
                f"  Could not find a valid FDR cell within 20 cells of "
                f"({_x:.1f}, {_y:.1f}). Aborting."
            )
            sys.exit(1)

        _nx, _ny = _fdr_trans * (_best_c + 0.5, _best_r + 0.5)
        logger.info(
            f"  Relocated pour point from ({_x:.1f}, {_y:.1f}) FDR=0 "
            f"-> ({_nx:.1f}, {_ny:.1f}) FDR={_best_fdr} FAC={_best_fac:,.0f}"
        )
        _fixed_points.append((_best_r, _best_c, _nx, _ny, _best_fdr))

    # Save the (possibly relocated) pour points as a shapefile.
    # Passing a vector directly to WBT Watershed is more reliable than a
    # raster — it sidesteps all int16/int32/float64 dtype hangs entirely.
    from shapely.geometry import Point as _ShapelyPoint
    _fixed_gdf = gpd.GeoDataFrame(
        {"POUR_ID": range(1, len(_fixed_points) + 1)},
        geometry=[_ShapelyPoint(_x, _y) for _, _, _x, _y, _ in _fixed_points],
        crs=primary.crs,
    )
    pour_shp = output_dir / "pourpoints_final.shp"
    _fixed_gdf[["POUR_ID", "geometry"]].to_file(str(pour_shp))
    logger.info(f"  Pour point shapefile written: {pour_shp.name} ({len(_fixed_gdf)} point(s))")

    # ------------------------------------------------------------------
    # Step 3 & 4: WBT watershed delineation -> vectorise -> area stats
    # ------------------------------------------------------------------
    logger.info("Step 3: Delineating watershed with WhiteboxTools...")

    watersheds_tif = output_dir / "watersheds.tif"
    watersheds_shp = output_dir / "watersheds.shp"

    gdf = delineate_and_vectorise(
        fdr_path, pour_shp, watersheds_tif, watersheds_shp, logger
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
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        (output_dir / f"pourpoints_final{ext}").unlink(missing_ok=True)

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
