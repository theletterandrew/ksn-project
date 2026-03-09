"""
topotoolbox_streams.py
----------------------
Uses pytopotoolbox to extract the stream network and main stem (trunk river)
for each drainage basin produced by delineate_and_clip_basins.py.

For each basin_XXXX/ folder:
  1. Load the clipped basin DEM (dem.tif)
  2. Compute flow routing (FlowObject)
  3. Extract the stream network (StreamObject)
  4. Isolate the largest connected component
  5. Extract the trunk river (main stem)
  6. Export the trunk as a GeoPackage to basin_XXXX/main_stem.gpkg
  7. Export the full stream network as a GeoPackage to basin_XXXX/stream_network.gpkg

USAGE:
    python topotoolbox_streams.py

Dependencies:
    pip install topotoolbox geopandas shapely fiona
"""

import logging
import sys
import time
from pathlib import Path

import fiona
import fiona.crs
import geopandas as gpd
from shapely.geometry import LineString, mapping

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

# =============================================================================
# END CONFIG
# =============================================================================


def setup_logging(basins_dir: Path) -> logging.Logger:
    log_path = basins_dir / "topotoolbox_streams.log"
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

    # Flatten all coordinate groups into a single ordered list
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

    # Build one LineString per segment, skipping any degenerate groups
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


def process_basin(basin_dir: Path, logger: logging.Logger) -> bool:
    """
    Run the full topotoolbox stream extraction pipeline for one basin.
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
        logger.info(f"  Stream network extracted")

        # ── Step 4: Largest connected component ───────────────────────────────
        logger.info("  Step 4: Isolating largest connected component...")
        s_main = s.klargestconncomps(1)

        # ── Step 5: Trunk river ────────────────────────────────────────────────
        logger.info("  Step 5: Extracting trunk river...")
        trunk = s_main.trunk()

        # Get CRS from the source DEM using rasterio
        import rasterio
        with rasterio.open(str(dem_path)) as src:
            out_crs = src.crs.to_wkt() if src.crs else None

        # ── Step 6: Export trunk ───────────────────────────────────────────────
        logger.info("  Step 6: Exporting trunk to GeoPackage...")
        export_trunk(trunk, basin_dir, out_crs, logger)

        # ── Step 7: Export full stream network ────────────────────────────────
        logger.info("  Step 7: Exporting full stream network to GeoPackage...")
        export_stream_network(s_main, basin_dir, out_crs, logger)

        elapsed = time.time() - start
        logger.info(f"  Done in {elapsed:.1f}s")
        return True

    except Exception as e:
        logger.error(f"  FAILED: {e}", exc_info=True)
        return False


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
    logger.info(f"topotoolbox stream extraction — {len(basin_dirs)} basin(s)")
    logger.info(f"Threshold : {THRESHOLD} pixels")
    logger.info(f"Basins dir: {BASINS_DIR}")
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
