# -*- coding: utf-8 -*-
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
  6. Export the trunk as a GeoPackage    -> basin_XXXX/main_stem.gpkg
  7. Export the full stream network      -> basin_XXXX/stream_network.gpkg
  8. Sample elevation, distance, chi, and ksn along the trunk at 50m intervals
  9. Export sampled metrics as points    -> basin_XXXX/ksn_chi_points.gpkg
 10. Export three profile plots (PNG):
       basin_XXXX/plot_elev_distance.png
       basin_XXXX/plot_elev_chi.png
       basin_XXXX/plot_elev_ksn.png

Chi is computed with a reference concavity (m/n) of 0.45.
Ksn is computed as slope x drainage_area^(m/n), smoothed with a moving
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
#   25,000 pixels  = ~0.1 km^2
#   250,000 pixels = ~1 km^2
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
A0 = config.A0

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
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)),
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
# =============================================================================
# KSN / CHI HELPERS
# =============================================================================

def moving_average(arr, window):
    """Apply a centred moving average, preserving array length."""
    if window < 2 or len(arr) < window:
        return arr.copy()
    padded = np.pad(arr, window // 2, mode="edge")
    smoothed = np.convolve(padded, np.ones(window) / window, mode="valid")
    return smoothed[:len(arr)]


# =============================================================================
# TRUNK SAMPLING
# =============================================================================

def sample_trunk(fd, trunk, dem, spacing, theta, a0, ksn_window, logger):
    """
    Sample trunk metrics using topotoolbox native methods:
      - fd.flow_accumulation()         : upstream contributing area (GridObject)
      - trunk.ezgetnal(dem)            : elevation NAL at each trunk node
      - trunk.chitransform(acc)        : chi NAL at each trunk node
      - trunk.ksn(dem, acc, theta)     : ksn NAL at each trunk node
      - trunk.distance()               : distance from mouth NAL (m)

    All NALs are co-indexed. Ksn uses the native topotoolbox implementation
    (slope/area normalisation) with optional minima imposition to avoid
    negative slopes. NALs are sorted mouth->headwater then resampled to
    uniform spacing for export.

    Returns a dict of uniformly spaced arrays, or None on failure.
    """
    try:
        # ── Native topotoolbox metrics ─────────────────────────────────────────
        acc      = fd.flow_accumulation()
        elev_nal = np.asarray(trunk.ezgetnal(dem),                          dtype=float)
        chi_nal  = np.asarray(trunk.chitransform(acc, mn=theta, a0=a0),    dtype=float)
        ksn_nal  = np.asarray(trunk.ksn(dem, acc, impose=True, theta=theta), dtype=float)
        dist_nal = np.asarray(trunk.upstream_distance(),                    dtype=float)
        acc_nal  = np.asarray(trunk.ezgetnal(acc),                          dtype=float)

        cellsize  = float(dem.cellsize)
        drain_nal = np.where(acc_nal > 0, acc_nal * (cellsize ** 2), cellsize ** 2)

        # ── Sort mouth -> headwater ────────────────────────────────────────────
        order     = np.argsort(dist_nal)
        dist_nal  = dist_nal[order]
        elev_nal  = elev_nal[order]
        chi_nal   = chi_nal[order]
        ksn_nal   = ksn_nal[order]
        drain_nal = drain_nal[order]

        # ── Remove duplicate distance values to avoid gradient divide-by-zero ──
        _, unique_idx = np.unique(dist_nal, return_index=True)
        dist_nal  = dist_nal[unique_idx]
        elev_nal  = elev_nal[unique_idx]
        chi_nal   = chi_nal[unique_idx]
        ksn_nal   = ksn_nal[unique_idx]
        drain_nal = drain_nal[unique_idx]

        # ── Smooth ksn with moving window ─────────────────────────────────────
        ksn_nal = moving_average(ksn_nal, ksn_window)

        # ── Slope from elevation gradient (for export only) ───────────────────
        slope_nal = np.clip(np.abs(np.gradient(elev_nal, dist_nal)), 1e-6, None)

        # ── XY coordinates aligned to dist_nal ────────────────────────────────
        # Always interpolate xy onto dist_nal using cumulative chord distance
        # along the trunk coordinate list. This is robust regardless of whether
        # node counts match after sorting and deduplication.
        coord_groups = trunk.xy()
        all_coords = []
        for group in coord_groups:
            all_coords.extend(group)
        coords = np.array(all_coords)

        dists_xy = np.zeros(len(coords))
        for i in range(1, len(coords)):
            dists_xy[i] = dists_xy[i-1] + np.hypot(
                coords[i,0] - coords[i-1,0],
                coords[i,1] - coords[i-1,1]
            )
        xs_nal = np.interp(dist_nal, dists_xy, coords[:,0])
        ys_nal = np.interp(dist_nal, dists_xy, coords[:,1])

        # ── Resample to uniform spacing ────────────────────────────────────────
        sample_dists = np.arange(0, dist_nal[-1], spacing)
        xs        = np.interp(sample_dists, dist_nal, xs_nal)
        ys        = np.interp(sample_dists, dist_nal, ys_nal)
        elevation = np.interp(sample_dists, dist_nal, elev_nal)
        chi       = np.interp(sample_dists, dist_nal, chi_nal)
        drain     = np.interp(sample_dists, dist_nal, drain_nal)
        slope     = np.interp(sample_dists, dist_nal, slope_nal)
        ksn       = np.interp(sample_dists, dist_nal, ksn_nal)

        n_nan_elev = int(np.sum(np.isnan(elevation)))
        n_nan_ksn  = int(np.sum(np.isnan(ksn)))
        logger.info(f"  Sampled {len(sample_dists)} points "
                    f"({dist_nal[-1]/1000:.2f} km trunk, {spacing:.0f} m spacing) "
                    f"| NaN elev: {n_nan_elev}, NaN ksn: {n_nan_ksn}")

        return {
            "x": xs, "y": ys,
            "distance_m":       sample_dists,
            "elevation_m":      elevation,
            "drainage_area_m2": drain,
            "slope":            slope,
            "chi":              chi,
            "ksn":              ksn,
        }

    except Exception as e:
        logger.error(f"  Trunk sampling failed: {e}", exc_info=True)
        return None


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

    logger.info(f"  Exported {len(features)} points -> {out_path}")


def export_plots(data: dict, basin_dir: Path, basin_name: str,
                 logger: logging.Logger) -> None:
    """
    Generate and save three elevation profile plots.

    Elevation vs Distance and Elevation vs Chi are line plots — both x-axes
    are monotonically increasing along the profile so a connected line makes
    sense.

    Elevation vs Ksn uses distance as the x-axis with the line coloured by
    ksn value. Plotting ksn directly on the x-axis produces a tangled line
    because ksn is not monotonic, making the plot unreadable.
    """
    dist_km = data["distance_m"] / 1000.0
    elev    = data["elevation_m"]
    chi     = data["chi"]
    ksn     = data["ksn"]

    if np.all(np.isnan(elev)):
        logger.warning("  Skipping all plots — elevation data is all NaN")
        return

    # ── Elevation vs Distance ──────────────────────────────────────────────────
    for x_vals, xlabel, filename, title in [
        (dist_km, "Distance from mouth (km)",
         "plot_elev_distance.png", f"{basin_name} - Elevation vs Distance"),
        (chi,     f"Chi (m/n = {THETA_REF}, A0 = {A0} m^2)",
         "plot_elev_chi.png",      f"{basin_name} - Elevation vs Chi"),
    ]:
        if np.all(np.isnan(x_vals)):
            logger.warning(f"  Skipping {filename} — x data is all NaN")
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x_vals, elev, color="#2563eb", linewidth=1.2)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Elevation (m)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        out_path = basin_dir / filename
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        logger.info(f"  Saved plot -> {out_path}")

    # ── Elevation vs Distance, coloured by Ksn ────────────────────────────────
    # Ksn is not monotonic so plotting it on the x-axis produces a tangled
    # line. Instead we plot distance on x, elevation on y, and colour each
    # segment by its ksn value using a LineCollection.
    if not np.all(np.isnan(ksn)):
        from matplotlib.collections import LineCollection
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        fig, ax = plt.subplots(figsize=(9, 5))

        # Build segments: each segment connects adjacent sample points
        points  = np.array([dist_km, elev]).T.reshape(-1, 1, 2)
        segs    = np.concatenate([points[:-1], points[1:]], axis=1)
        ksn_mid = (ksn[:-1] + ksn[1:]) / 2.0   # ksn value for each segment

        norm = Normalize(vmin=np.nanpercentile(ksn, 5),
                         vmax=np.nanpercentile(ksn, 95))
        lc = LineCollection(segs, cmap="plasma", norm=norm, linewidth=2)
        lc.set_array(ksn_mid)
        ax.add_collection(lc)

        ax.set_xlim(dist_km.min(), dist_km.max())
        ax.set_ylim(elev.min() - 20, elev.max() + 20)
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap="plasma"), ax=ax)
        cbar.set_label(f"Ksn (smoothed, window = {KSN_WINDOW} pts x {POINT_SPACING_M} m)",
                       fontsize=10)
        ax.set_xlabel("Distance from mouth (km)", fontsize=11)
        ax.set_ylabel("Elevation (m)", fontsize=11)
        ax.set_title(f"{basin_name} - Elevation Profile coloured by Ksn",
                     fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        out_path = basin_dir / "plot_elev_ksn.png"
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        logger.info(f"  Saved plot -> {out_path}")
    else:
        logger.warning("  Skipping plot_elev_ksn.png — ksn data is all NaN")


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
        logger.info("  Step 8: Sampling trunk metrics...")
        data = sample_trunk(
            fd, trunk, dem,
            spacing=POINT_SPACING_M, theta=THETA_REF, a0=A0,
            ksn_window=KSN_WINDOW, logger=logger,
        )

        if data is None:
            logger.error("  Trunk sampling failed — ksn/chi outputs skipped.")
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
