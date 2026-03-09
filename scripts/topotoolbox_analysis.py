"""
topotoolbox_analysis.py
-----------------------
Uses pytopotoolbox to extract the stream network, main stem (trunk river),
and longitudinal profile metrics (ksn, chi) for each drainage basin produced
by delineate_and_clip_basins.py.

For each basin_XXXX/ folder:
  1. Load the clipped basin DEM (dem.tif)
  2. Compute flow routing (FlowObject)
  3. Extract the stream network (StreamObject)
  4. Isolate the largest connected component
  5. Extract the trunk river (main stem)
  6. Export the trunk as a GeoPackage    → basin_XXXX/main_stem.gpkg
  7. Export the full stream network      → basin_XXXX/stream_network.gpkg
  8. Sample elevation, distance, chi, and ksn along the trunk at 50m intervals
  9. Export sampled metrics as points    → basin_XXXX/ksn_chi_points.gpkg
 10. Export three profile plots (PNG):
       basin_XXXX/plot_elev_distance.png
       basin_XXXX/plot_elev_chi.png
       basin_XXXX/plot_elev_ksn.png

Chi is computed with a reference concavity (m/n) of 0.45.
Ksn is computed as slope × drainage_area^(m/n), smoothed with a moving
window average along the trunk.

USAGE:
    python topotoolbox_analysis.py

Dependencies:
    pip install topotoolbox geopandas shapely fiona matplotlib numpy rasterio
"""

import logging
import sys
import time
from pathlib import Path

import fiona
import fiona.crs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from shapely.geometry import LineString, Point, mapping

import topotoolbox as tt

# Calculate the path to the project root (one level up from scripts/)
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

# =============================================================================
# CONFIG
# =============================================================================

BASINS_DIR = Path(config.DATA_BASINS)

# Stream initiation threshold in pixels. At 2m resolution:
#   25,000 pixels  = ~0.1 km²
#   250,000 pixels = ~1 km²
# Adjust until the extracted network looks right for your basins.
THRESHOLD = config.STREAM_THRESHOLD

# Reference concavity index (m/n) for chi integration
THETA_REF = config.REFERENCE_CONCAVITY

# Point spacing along trunk for ksn/chi export (metres)
POINT_SPACING_M = config.SAMPLE_DISTANCE

# Moving window size for ksn smoothing (number of sample points).
# At 50m spacing, 10 points = 500m window.
KSN_WINDOW = config.SMOOTHING_WINDOW

# Reference drainage area for chi (A0). Set to 1 m² so chi has units of
# metres, making it directly comparable across basins.
A0 = 1.0

# =============================================================================
# END CONFIG
# =============================================================================


def setup_logging(basins_dir: Path) -> logging.Logger:
    log_path = basins_dir / "topotoolbox_analysis.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# =============================================================================
# STREAM NETWORK EXPORTS
# =============================================================================

def export_trunk(trunk, basin_dir: Path, out_crs, logger: logging.Logger) -> None:
    """
    Export a StreamObject trunk to a GeoPackage as a single LineString.

    trunk.xy() returns a list of sublists, one per tributary segment.
    For the trunk there is typically one sublist — we chain all coordinates
    into a single LineString ordered mouth->headwater.
    """
    coord_groups = trunk.xy()

    if not coord_groups:
        logger.warning("  trunk.xy() returned no coordinates — skipping export.")
        return

    all_coords = []
    for group in coord_groups:
        all_coords.extend(group)

    if len(all_coords) < 2:
        logger.warning("  Trunk has fewer than 2 coordinate points — skipping export.")
        return

    line = LineString(all_coords)
    length_m = line.length
    logger.info(f"  Trunk length: {length_m:.1f} m  ({length_m / 1000:.2f} km)")

    out_path = basin_dir / "main_stem.gpkg"
    if out_path.exists():
        out_path.unlink()

    schema = {
        "geometry": "LineString",
        "properties": {
            "length_m":   "float",
            "n_vertices": "int",
        },
    }

    with fiona.open(
        str(out_path), mode="w", driver="GPKG",
        schema=schema, crs=out_crs, layer="main_stem",
    ) as dst:
        dst.write({
            "geometry": mapping(line),
            "properties": {
                "length_m":   round(length_m, 2),
                "n_vertices": len(all_coords),
            },
        })

    logger.info(f"  Written to: {out_path}")


def export_stream_network(stream, basin_dir: Path, out_crs, logger: logging.Logger) -> None:
    """
    Export a StreamObject's full network to a GeoPackage.

    stream.xy() returns one sublist of coordinates per segment (i.e. per
    tributary branch). Each sublist becomes its own LineString feature so
    that the network retains its branching structure in GIS.
    """
    coord_groups = stream.xy()

    if not coord_groups:
        logger.warning("  stream.xy() returned no coordinates — skipping network export.")
        return

    features = []
    for group in coord_groups:
        if len(group) < 2:
            continue
        line = LineString(group)
        features.append({
            "geometry": mapping(line),
            "properties": {
                "length_m":   round(line.length, 2),
                "n_vertices": len(group),
            },
        })

    if not features:
        logger.warning("  No valid segments found — skipping network export.")
        return

    logger.info(f"  Stream network segments: {len(features)}")

    out_path = basin_dir / "stream_network.gpkg"
    if out_path.exists():
        out_path.unlink()

    schema = {
        "geometry": "LineString",
        "properties": {
            "length_m":   "float",
            "n_vertices": "int",
        },
    }

    with fiona.open(
        str(out_path), mode="w", driver="GPKG",
        schema=schema, crs=out_crs, layer="stream_network",
    ) as dst:
        dst.writerecords(features)

    logger.info(f"  Written to: {out_path}")


# =============================================================================
# KSN / CHI HELPERS
# =============================================================================

def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """Apply a centred moving average, preserving array length."""
    if window < 2 or len(arr) < window:
        return arr.copy()
    padded = np.pad(arr, window // 2, mode="edge")
    smoothed = np.convolve(padded, np.ones(window) / window, mode="valid")
    return smoothed[:len(arr)]


def compute_chi(distance_m: np.ndarray, drainage_area_m2: np.ndarray,
                theta: float, a0: float) -> np.ndarray:
    """
    Numerically integrate chi along the trunk from mouth to headwater.

    chi(x) = integral from mouth to x of (A0 / A(x'))^theta dx'

    distance_m and drainage_area_m2 must be ordered mouth -> headwater.
    Uses the trapezoidal rule.
    """
    integrand = (a0 / drainage_area_m2) ** theta
    dx = np.diff(distance_m)
    mid = (integrand[:-1] + integrand[1:]) / 2.0
    chi = np.zeros(len(distance_m))
    chi[1:] = np.cumsum(mid * dx)
    return chi


def compute_ksn(slope: np.ndarray, drainage_area_m2: np.ndarray,
                theta: float, window: int) -> np.ndarray:
    """
    Compute ksn = slope / (A^theta), then smooth with a moving window.
    """
    safe_area = np.where(drainage_area_m2 > 0, drainage_area_m2, np.nan)
    ksn_raw = slope / (safe_area ** theta)
    ksn_raw = np.nan_to_num(ksn_raw, nan=0.0)
    return moving_average(ksn_raw, window)


# =============================================================================
# TRUNK SAMPLING
# =============================================================================

def sample_trunk_topotoolbox(fd, trunk, dem, spacing: float,
                              theta: float, a0: float,
                              ksn_window: int, logger: logging.Logger) -> dict | None:
    """
    Sample trunk metrics using topotoolbox's native flow accumulation where
    available, falling back to a linear drainage area approximation if not.

    Returns a dict of arrays or None on failure.
    """
    try:
        coord_groups = trunk.xy()
        if not coord_groups:
            logger.warning("  trunk.xy() returned no coordinates.")
            return None

        all_coords = []
        for group in coord_groups:
            all_coords.extend(group)
        coords = np.array(all_coords)

        if len(coords) < 2:
            return None

        # ── Distance along trunk ───────────────────────────────────────────────
        dists_raw = np.zeros(len(coords))
        for i in range(1, len(coords)):
            dists_raw[i] = dists_raw[i - 1] + np.hypot(
                coords[i, 0] - coords[i - 1, 0],
                coords[i, 1] - coords[i - 1, 1]
            )

        sample_dists = np.arange(0, dists_raw[-1], spacing)
        xs = np.interp(sample_dists, dists_raw, coords[:, 0])
        ys = np.interp(sample_dists, dists_raw, coords[:, 1])

        # ── Elevation ─────────────────────────────────────────────────────────
        cellsize = float(dem.cellsize)
        origin_x = float(dem.georef.get("west", 0))
        origin_y = float(dem.georef.get("north", 0))

        def xy_to_rc(x, y):
            col = ((x - origin_x) / cellsize).astype(int)
            row = ((origin_y - y) / cellsize).astype(int)
            return row, col

        rows, cols = xy_to_rc(xs, ys)
        rows = np.clip(rows, 0, dem.z.shape[0] - 1)
        cols = np.clip(cols, 0, dem.z.shape[1] - 1)
        elevation = dem.z[rows, cols].astype(float)

        # ── Drainage area ──────────────────────────────────────────────────────
        try:
            acc = tt.flowacc(fd)
            acc_grid = np.array(acc.z).astype(float)
            drainage_area = acc_grid[rows, cols] * (cellsize ** 2)
            drainage_area = np.where(drainage_area <= 0, cellsize ** 2, drainage_area)
            logger.info("  Using topotoolbox flow accumulation for drainage area.")
        except Exception:
            n = len(sample_dists)
            drainage_area = np.linspace(n * cellsize ** 2, cellsize ** 2, n)
            logger.info("  Using linear drainage area approximation.")

        # ── Slope, chi, ksn ───────────────────────────────────────────────────
        slope = np.clip(np.abs(np.gradient(elevation, sample_dists)), 1e-6, None)
        chi = compute_chi(sample_dists, drainage_area, theta, a0)
        ksn = compute_ksn(slope, drainage_area, theta, ksn_window)

        return {
            "x": xs, "y": ys,
            "distance_m":       sample_dists,
            "elevation_m":      elevation,
            "drainage_area_m2": drainage_area,
            "slope":            slope,
            "chi":              chi,
            "ksn":              ksn,
        }

    except Exception as e:
        logger.error(f"  topotoolbox sampling failed: {e}", exc_info=True)
        return None


def sample_trunk_numpy(trunk_xy, dem_path: Path, cellsize: float,
                       spacing: float, theta: float, a0: float,
                       ksn_window: int, logger: logging.Logger) -> dict | None:
    """
    Pure-numpy fallback for trunk sampling when topotoolbox methods fail.
    Uses a linear drainage area approximation.

    Returns a dict of arrays or None on failure.
    """
    try:
        all_coords = []
        for group in trunk_xy:
            all_coords.extend(group)

        if len(all_coords) < 2:
            return None

        coords = np.array(all_coords)

        dists_raw = np.zeros(len(coords))
        for i in range(1, len(coords)):
            dists_raw[i] = dists_raw[i - 1] + np.hypot(
                coords[i, 0] - coords[i - 1, 0],
                coords[i, 1] - coords[i - 1, 1]
            )

        sample_dists = np.arange(0, dists_raw[-1], spacing)
        xs = np.interp(sample_dists, dists_raw, coords[:, 0])
        ys = np.interp(sample_dists, dists_raw, coords[:, 1])

        with rasterio.open(str(dem_path)) as src:
            transform = src.transform
            dem_data = src.read(1).astype(float)
            nodata = src.nodata
            if nodata is not None:
                dem_data[dem_data == nodata] = np.nan

            def xy_to_rc(x, y):
                col = (x - transform.c) / transform.a
                row = (y - transform.f) / transform.e
                return row.astype(int), col.astype(int)

            rows, cols = xy_to_rc(xs, ys)
            rows = np.clip(rows, 0, dem_data.shape[0] - 1)
            cols = np.clip(cols, 0, dem_data.shape[1] - 1)
            elevation = dem_data[rows, cols]

        n = len(sample_dists)
        drainage_area = np.linspace(n * cellsize ** 2, cellsize ** 2, n)

        slope = np.clip(np.abs(np.gradient(elevation, sample_dists)), 1e-6, None)
        chi = compute_chi(sample_dists, drainage_area, theta, a0)
        ksn = compute_ksn(slope, drainage_area, theta, ksn_window)

        return {
            "x": xs, "y": ys,
            "distance_m":       sample_dists,
            "elevation_m":      elevation,
            "drainage_area_m2": drainage_area,
            "slope":            slope,
            "chi":              chi,
            "ksn":              ksn,
        }

    except Exception as e:
        logger.error(f"  numpy fallback failed: {e}", exc_info=True)
        return None


# =============================================================================
# KSN / CHI EXPORTS
# =============================================================================

def export_ksn_chi_points(data: dict, basin_dir: Path, out_crs,
                           logger: logging.Logger) -> None:
    """Export sampled trunk metrics as GeoPackage points."""
    out_path = basin_dir / "ksn_chi_points.gpkg"
    if out_path.exists():
        out_path.unlink()

    schema = {
        "geometry": "Point",
        "properties": {
            "distance_m":       "float",
            "elevation_m":      "float",
            "drainage_area_m2": "float",
            "slope":            "float",
            "chi":              "float",
            "ksn":              "float",
        },
    }

    features = [
        {
            "geometry": mapping(Point(float(data["x"][i]), float(data["y"][i]))),
            "properties": {
                "distance_m":       round(float(data["distance_m"][i]), 2),
                "elevation_m":      round(float(data["elevation_m"][i]), 3),
                "drainage_area_m2": round(float(data["drainage_area_m2"][i]), 2),
                "slope":            round(float(data["slope"][i]), 6),
                "chi":              round(float(data["chi"][i]), 4),
                "ksn":              round(float(data["ksn"][i]), 4),
            },
        }
        for i in range(len(data["x"]))
    ]

    with fiona.open(
        str(out_path), mode="w", driver="GPKG",
        schema=schema, crs=out_crs, layer="ksn_chi_points",
    ) as dst:
        dst.writerecords(features)

    logger.info(f"  Exported {len(features)} points → {out_path}")


def export_plots(data: dict, basin_dir: Path, basin_name: str,
                 logger: logging.Logger) -> None:
    """Generate and save the three elevation profile plots."""
    dist_km = data["distance_m"] / 1000.0
    elev    = data["elevation_m"]
    chi     = data["chi"]
    ksn     = data["ksn"]

    plot_specs = [
        {
            "x":        dist_km,
            "xlabel":   "Distance from mouth (km)",
            "filename": "plot_elev_distance.png",
            "title":    f"{basin_name} — Elevation vs Distance",
        },
        {
            "x":        chi,
            "xlabel":   f"Chi (m/n = {THETA_REF}, A₀ = {A0} m²)",
            "filename": "plot_elev_chi.png",
            "title":    f"{basin_name} — Elevation vs Chi",
        },
        {
            "x":        ksn,
            "xlabel":   f"Ksn (smoothed, window = {KSN_WINDOW} pts × {POINT_SPACING_M} m)",
            "filename": "plot_elev_ksn.png",
            "title":    f"{basin_name} — Elevation vs Ksn",
        },
    ]

    for spec in plot_specs:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(spec["x"], elev, color="#2563eb", linewidth=1.2)
        ax.set_xlabel(spec["xlabel"], fontsize=11)
        ax.set_ylabel("Elevation (m)", fontsize=11)
        ax.set_title(spec["title"], fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        out_path = basin_dir / spec["filename"]
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        logger.info(f"  Saved plot → {out_path}")


# =============================================================================
# PER-BASIN PIPELINE
# =============================================================================

def process_basin(basin_dir: Path, logger: logging.Logger) -> bool:
    """
    Run the full analysis pipeline for one basin.
    Flow objects are built once and shared across all steps.
    Returns True on success, False on failure.
    """
    dem_path = basin_dir / "dem.tif"
    if not dem_path.exists():
        logger.error(f"  dem.tif not found in {basin_dir}")
        return False

    start = time.time()

    try:
        # ── Step 1: Load DEM ───────────────────────────────────────────────────
        logger.info("  Step 1: Loading DEM...")
        dem = tt.read_tif(str(dem_path))
        logger.info(f"  DEM shape: {dem.shape}  resolution: {dem.cellsize:.1f} m")

        # ── Step 2: Flow routing ───────────────────────────────────────────────
        logger.info("  Step 2: Computing flow routing (FlowObject)...")
        fd = tt.FlowObject(dem)

        # ── Step 3: Stream network ─────────────────────────────────────────────
        logger.info(f"  Step 3: Extracting stream network (threshold={THRESHOLD} pixels)...")
        s = tt.StreamObject(fd, threshold=THRESHOLD, units="pixels")

        # ── Step 4: Largest connected component ───────────────────────────────
        logger.info("  Step 4: Isolating largest connected component...")
        s_main = s.klargestconncomps(1)

        # ── Step 5: Trunk river ────────────────────────────────────────────────
        logger.info("  Step 5: Extracting trunk river...")
        trunk = s_main.trunk()

        # Get CRS and cell size once for all subsequent steps
        with rasterio.open(str(dem_path)) as src:
            out_crs  = src.crs.to_wkt() if src.crs else None
            cellsize = src.res[0]

        # ── Step 6: Export trunk ───────────────────────────────────────────────
        logger.info("  Step 6: Exporting trunk to GeoPackage...")
        export_trunk(trunk, basin_dir, out_crs, logger)

        # ── Step 7: Export full stream network ────────────────────────────────
        logger.info("  Step 7: Exporting full stream network to GeoPackage...")
        export_stream_network(s_main, basin_dir, out_crs, logger)

        # ── Step 8: Sample trunk metrics ──────────────────────────────────────
        logger.info("  Step 8: Sampling trunk metrics (topotoolbox)...")
        data = sample_trunk_topotoolbox(
            fd, trunk, dem,
            spacing=POINT_SPACING_M, theta=THETA_REF, a0=A0,
            ksn_window=KSN_WINDOW, logger=logger,
        )

        if data is None:
            logger.warning("  topotoolbox sampling failed — trying numpy fallback...")
            data = sample_trunk_numpy(
                trunk.xy(), dem_path, cellsize,
                spacing=POINT_SPACING_M, theta=THETA_REF, a0=A0,
                ksn_window=KSN_WINDOW, logger=logger,
            )

        if data is None:
            logger.error("  Both sampling methods failed — ksn/chi outputs skipped.")
            # Still count as partial success since network exports completed
            return True

        logger.info(f"  {len(data['x'])} sample points along trunk")

        # ── Step 9: Export ksn/chi points ─────────────────────────────────────
        logger.info("  Step 9: Exporting ksn/chi points to GeoPackage...")
        export_ksn_chi_points(data, basin_dir, out_crs, logger)

        # ── Step 10: Export plots ──────────────────────────────────────────────
        logger.info("  Step 10: Generating profile plots...")
        export_plots(data, basin_dir, basin_dir.name, logger)

        elapsed = time.time() - start
        logger.info(f"  Done in {elapsed:.1f}s")
        return True

    except Exception as e:
        logger.error(f"  FAILED: {e}", exc_info=True)
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not BASINS_DIR.exists():
        print(f"ERROR: Basins directory not found: {BASINS_DIR}")
        sys.exit(1)

    logger = setup_logging(BASINS_DIR)

    basin_dirs = sorted([
        d for d in BASINS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("basin_") and (d / "dem.tif").exists()
    ])

    if not basin_dirs:
        logger.error(f"No basin_XXXX/dem.tif found in {BASINS_DIR}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"topotoolbox analysis — {len(basin_dirs)} basin(s)")
    logger.info(f"Threshold       : {THRESHOLD} pixels")
    logger.info(f"Concavity (m/n) : {THETA_REF}")
    logger.info(f"Point spacing   : {POINT_SPACING_M} m")
    logger.info(f"Ksn window      : {KSN_WINDOW} pts ({KSN_WINDOW * POINT_SPACING_M} m)")
    logger.info(f"Basins dir      : {BASINS_DIR}")
    logger.info("=" * 60)

    total_start = time.time()
    succeeded, failed = [], []

    for basin_dir in basin_dirs:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Processing: {basin_dir.name}")
        logger.info("=" * 60)

        ok = process_basin(basin_dir, logger)
        if ok:
            succeeded.append(basin_dir.name)
        else:
            failed.append(basin_dir.name)

    elapsed_total = time.time() - total_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("ALL BASINS COMPLETE")
    logger.info(f"  Succeeded : {len(succeeded)}")
    logger.info(f"  Failed    : {len(failed)}")
    if failed:
        for name in failed:
            logger.warning(f"    FAILED: {name}")
    logger.info(f"  Total time: {elapsed_total / 60:.1f} minutes")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
