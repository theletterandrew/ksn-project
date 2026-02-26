"""
stream_extraction_wbt.py
------------------------
Extracts a fully connected stream network from WhiteboxTools flow
accumulation output. Converts the thresholded stream raster to vector
polylines using open-source Python GIS libraries (no arcpy required).

USAGE:
    1. Install dependencies:
       pip install rasterio numpy scipy scikit-image fiona shapely

    2. Edit the paths and threshold in the CONFIG section below.

    3. Run:
       python stream_extraction_wbt.py

Requirements:
    - rasterio
    - numpy
    - scipy
    - scikit-image
    - fiona
    - shapely
    - Completed wbt_hydrology.py first
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import fiona
from fiona.crs import from_epsg
from shapely.geometry import mapping, LineString, MultiLineString
from shapely.ops import unary_union
from scipy import ndimage
from skimage.morphology import skeletonize

# Calculate the path to the project root (one level up from scripts/)
root_dir = Path(__file__).resolve().parent.parent

# Add the root directory to sys.path so Python can find config.py
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

# =============================================================================
# CONFIG — Edit these before running
# =============================================================================

WBT_DIR     = config.DATA_SCRATCH_WBT   # Folder with WBT outputs
OUTPUT_DIR  = config.DATA_STREAMS       # Output folder for streams

FAC_FILE    = "flow_accumulation.tif"   # Flow accumulation from WBT
FDR_FILE    = "flow_direction.tif"      # Flow direction from WBT
OUTPUT_FILE = "streams_connected.shp"   # Output stream network

# Drainage area threshold
# Since this is from the full continuous mosaic, flow accumulates across
# the entire study area without tile boundary resets. Higher thresholds
# are now appropriate to avoid overly dense networks.
# At 2m resolution:
#   500,000 cells   = ~2 km²   (dense network)
#   1,000,000 cells = ~4 km²   (moderate)
#   2,500,000 cells = ~10 km²  (major channels only)
THRESHOLD = config.STREAM_THRESHOLD  # cells (~4 km² at 2m resolution)

# =============================================================================
# END CONFIG — No edits needed below this line
# =============================================================================


def setup_logging(output_dir: Path) -> logging.Logger:
    log_path = output_dir / "stream_extraction_wbt.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def pixel_to_coord(row, col, transform):
    """Convert raster pixel (row, col) to real-world (x, y) coordinates."""
    x, y = rasterio.transform.xy(transform, row, col)
    return x, y


def trace_stream_segments(stream_mask, transform):
    """
    Trace connected stream pixels into polyline segments.

    Strategy:
      1. Skeletonize the binary stream mask to single-pixel-wide lines.
      2. Label connected components.
      3. For each component, order the pixels along the path and build
         a LineString from their real-world coordinates.

    Returns a list of Shapely LineString geometries.
    """
    # Skeletonize to ensure single-pixel-wide centerlines
    skeleton = skeletonize(stream_mask)

    # Label 8-connected components
    struct = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    labeled, num_features = ndimage.label(skeleton, structure=struct)

    lines = []

    for label_id in range(1, num_features + 1):
        component = np.argwhere(labeled == label_id)

        if len(component) < 2:
            # Single isolated pixel — skip (can't form a line)
            continue

        if len(component) == 2:
            # Exactly two pixels — simple segment
            coords = [pixel_to_coord(r, c, transform) for r, c in component]
            lines.append(LineString(coords))
            continue

        # Order pixels by traversal from one endpoint to the other.
        # An endpoint is a pixel with exactly one neighbour in the skeleton.
        ordered = _order_pixels(component, skeleton)
        if ordered is None or len(ordered) < 2:
            continue

        coords = [pixel_to_coord(r, c, transform) for r, c in ordered]
        lines.append(LineString(coords))

    return lines


def _order_pixels(pixels, skeleton):
    """
    Order an array of (row, col) pixels from one end of a skeleton branch
    to the other using a greedy nearest-neighbour walk starting from an
    endpoint (degree-1 pixel).
    """
    pixel_set = {(r, c) for r, c in pixels}

    # Find endpoints: pixels with only one 8-connected neighbour in the set
    def degree(r, c):
        count = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr, dc) == (0, 0):
                    continue
                if (r + dr, c + dc) in pixel_set:
                    count += 1
        return count

    endpoints = [(r, c) for r, c in pixel_set if degree(r, c) == 1]

    # Start from an endpoint if available, otherwise any pixel
    start = endpoints[0] if endpoints else pixels[0].tolist()

    visited = []
    current = tuple(start)
    seen = set()

    while current in pixel_set and current not in seen:
        visited.append(current)
        seen.add(current)

        r, c = current
        neighbours = [
            (r + dr, c + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
            and (r + dr, c + dc) in pixel_set
            and (r + dr, c + dc) not in seen
        ]

        if not neighbours:
            break
        current = neighbours[0]

    return visited if len(visited) >= 2 else None


def main():
    wbt_dir    = Path(WBT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    fac_path = wbt_dir / FAC_FILE
    fdr_path = wbt_dir / FDR_FILE
    out_shp  = output_dir / OUTPUT_FILE

    # Validate inputs
    if not fac_path.exists():
        logger.error(f"Flow accumulation not found: {fac_path}")
        logger.error("Run wbt_hydrology.py first.")
        sys.exit(1)
    if not fdr_path.exists():
        logger.error(f"Flow direction not found: {fdr_path}")
        logger.error("Run wbt_hydrology.py first.")
        sys.exit(1)

    logger.info(f"Threshold: {THRESHOLD:,} cells (~{THRESHOLD * 4 / 1e6:.1f} km² at 2m)")
    logger.info("-" * 60)

    start_time = time.time()

    try:
        # --- Step 1: Read flow accumulation and apply threshold ---
        logger.info("Reading flow accumulation raster...")
        with rasterio.open(str(fac_path)) as fac_ds:
            fac_data  = fac_ds.read(1).astype(np.float64)
            transform = fac_ds.transform
            crs       = fac_ds.crs
            nodata    = fac_ds.nodata

        # Mask nodata pixels before thresholding
        if nodata is not None:
            valid_mask = fac_data != nodata
        else:
            valid_mask = np.ones_like(fac_data, dtype=bool)

        logger.info("Applying threshold to flow accumulation...")
        stream_mask = (fac_data >= THRESHOLD) & valid_mask
        logger.info(f"  Stream pixels above threshold: {stream_mask.sum():,}")

        if stream_mask.sum() == 0:
            logger.error("No stream pixels found at this threshold. "
                         "Lower THRESHOLD or check input data.")
            sys.exit(1)

        # --- Step 2: Trace stream pixels into vector polylines ---
        logger.info("Tracing stream network into polylines...")
        logger.info("  (This may take several minutes for large study areas)")

        lines = trace_stream_segments(stream_mask, transform)
        logger.info(f"  Polyline segments created: {len(lines):,}")

        if not lines:
            logger.error("No polyline segments were created. "
                         "Check input data and threshold.")
            sys.exit(1)

        # --- Step 3: Write to shapefile ---
        logger.info(f"Writing output shapefile: {out_shp}")

        schema = {
            "geometry": "LineString",
            "properties": {"seg_id": "int"}
        }

        # Use the raster CRS for the output; fall back to EPSG:4326 if absent
        if crs is not None:
            out_crs = crs.to_dict()
        else:
            logger.warning("No CRS found in flow accumulation raster. "
                           "Output will have no projection.")
            out_crs = {}

        with fiona.open(
            str(out_shp),
            mode="w",
            driver="ESRI Shapefile",
            schema=schema,
            crs=out_crs
        ) as dst:
            for i, line in enumerate(lines, start=1):
                dst.write({
                    "geometry": mapping(line),
                    "properties": {"seg_id": i}
                })

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("COMPLETE")
        logger.info(f"  Output        : {out_shp}")
        logger.info(f"  Stream count  : {len(lines):,} segments")
        logger.info(f"  Total time    : {elapsed / 60:.1f} minutes")
        logger.info("")
        logger.info("Load streams_connected.shp in ArcGIS Pro or QGIS to verify.")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
