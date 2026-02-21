"""
batchdownload.py
----------------
Downloads ground-classified LiDAR tiles from a USGS EPT dataset by directly
traversing the EPT quadtree hierarchy and fetching LAZ node files.

Replaces PDAL with requests + laspy, preserving identical output:
  - Spatial subsetting via bounding box
  - Ground-only filtering (Classification=2)
  - LAZ-compressed output

EPT format reference: https://entwine.io/entwine-point-tile.html
"""

import io
import sys
import json
import time
import requests
import laspy
import numpy as np
from pathlib import Path
from shapely.geometry import box

# Project root (one level up from scripts/)
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config


# ---------------------------------------------------------------------------
# EPT Helpers
# ---------------------------------------------------------------------------

def fetch_json(url: str) -> dict:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def node_bounds(ept_bounds: list, d: int, x: int, y: int) -> tuple:
    """
    Calculate the spatial bounds of an EPT node given its depth/x/y address.
    EPT bounds are [minx, miny, minz, maxx, maxy, maxz].
    We only care about x/y for 2D intersection testing.
    Returns (minx, miny, maxx, maxy).
    """
    minx, miny, _, maxx, maxy, _ = ept_bounds
    step_x = (maxx - minx) / (2 ** d)
    step_y = (maxy - miny) / (2 ** d)
    return (
        minx + x * step_x,
        miny + y * step_y,
        minx + (x + 1) * step_x,
        miny + (y + 1) * step_y,
    )


def boxes_intersect(a: tuple, b: tuple) -> bool:
    """Check if two (minx, miny, maxx, maxy) boxes intersect."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def collect_nodes(
    hierarchy: dict,
    ept_bounds: list,
    query_box: tuple,
    base_url: str,
    d: int = 0,
    x: int = 0,
    y: int = 0,
    z: int = 0,
    nodes: list = None
) -> list:
    """
    Recursively traverse the EPT hierarchy to collect all node keys
    whose bounds intersect the query bounding box.

    Fetches sub-hierarchy JSON pages as needed (EPT splits large hierarchies
    into separate JSON files at certain depths).

    Note: The root node (0-0-0-0) is often absent from the hierarchy JSON
    itself — its existence is implied by ept.json. We handle this by treating
    the root as always present and starting recursion from its children.
    """
    if nodes is None:
        nodes = []

    key = f"{d}-{x}-{y}-{z}"

    # Root node is implied — skip the point count check for it
    # For all other nodes, check if they exist in the hierarchy
    if d > 0:
        point_count = hierarchy.get(key)
        if point_count is None or point_count == 0:
            return nodes  # Node doesn't exist or has no points

        nb = node_bounds(ept_bounds, d, x, y)
        if not boxes_intersect(nb, query_box):
            return nodes  # Node doesn't overlap our tile

        if point_count == -1:
            # Children are in a separate hierarchy page — fetch it
            sub_url = f"{base_url}/ept-hierarchy/{key}.json"
            try:
                sub_hierarchy = fetch_json(sub_url)
                hierarchy.update(sub_hierarchy)
            except Exception:
                return nodes

        nodes.append(key)
    else:
        # Root: always check spatial overlap but don't require it in hierarchy
        nb = node_bounds(ept_bounds, d, x, y)
        if not boxes_intersect(nb, query_box):
            return nodes

    # Recurse into children
    for dx in range(2):
        for dy in range(2):
            collect_nodes(
                hierarchy, ept_bounds, query_box, base_url,
                d + 1, x * 2 + dx, y * 2 + dy, z,
                nodes
            )

    return nodes


def download_node(base_url: str, key: str) -> bytes:
    """Download a single EPT LAZ node file and return its raw bytes."""
    url = f"{base_url}/ept-data/{key}.laz"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content


# ---------------------------------------------------------------------------
# Main Download Function
# ---------------------------------------------------------------------------

def run_download(tile_box, filename: str):
    """
    Downloads all EPT nodes intersecting tile_box, merges them, filters to
    ground points (Classification=2), and writes a LAZ file.
    """
    b = tile_box.bounds  # (minx, miny, maxx, maxy)
    query_box = (b[0], b[1], b[2], b[3])
    base_url = config.EPT_URL.rsplit("/", 1)[0]  # strip ept.json

    # --- Step 1: Load EPT metadata and root hierarchy ---
    ept_info = fetch_json(config.EPT_URL)
    ept_bounds = ept_info["bounds"]  # [minx, miny, minz, maxx, maxy, maxz]
    hierarchy = fetch_json(f"{base_url}/ept-hierarchy/0-0-0-0.json")

    # --- Step 2: Find all nodes that intersect our tile ---
    nodes = collect_nodes(hierarchy, ept_bounds, query_box, base_url)

    if not nodes:
        raise ValueError(
            f"No EPT nodes found intersecting bounds {query_box}. "
            "Check that BOUNDS_STR in config.py is within the dataset extent."
        )

    # --- Step 3: Download and merge all intersecting nodes ---
    all_points = []
    header = None

    for key in nodes:
        raw = download_node(base_url, key)
        with laspy.open(io.BytesIO(raw)) as reader:
            las = reader.read()
            if header is None:
                header = las.header
            all_points.append(las.points)

    merged_points = np.concatenate(all_points) if len(all_points) > 1 else all_points[0]

    # --- Step 4: Spatial clip to exact tile bounds ---
    # Nodes are coarser than our tile so we clip precisely.
    # Coordinates are stored as integers: real = offset + scale * int_val
    x_offset = header.offsets[0]
    y_offset = header.offsets[1]
    x_scale  = header.scales[0]
    y_scale  = header.scales[1]

    x_coords = x_offset + x_scale * merged_points.X.astype(np.float64)
    y_coords = y_offset + y_scale * merged_points.Y.astype(np.float64)

    spatial_mask = (
        (x_coords >= b[0]) & (x_coords <= b[2]) &
        (y_coords >= b[1]) & (y_coords <= b[3])
    )

    # --- Step 5: Filter to ground points only (Classification=2) ---
    ground_mask  = merged_points.classification == 2
    final_mask   = spatial_mask & ground_mask
    final_points = merged_points[final_mask]

    if len(final_points) == 0:
        raise ValueError(
            "No ground points found in this tile after filtering. "
            "The tile may be empty or outside the dataset extent."
        )

    # --- Step 6: Write filtered points as LAZ ---
    filtered = laspy.LasData(header=header)
    filtered.points = final_points

    with laspy.open(
        filename,
        mode="w",
        header=filtered.header,
        laz_backend=laspy.LazBackend.LazrsParallel
    ) as writer:
        writer.write_points(filtered.points)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)

    # Verify EPT endpoint is reachable before starting
    print(f"Verifying EPT endpoint: {config.EPT_URL}")
    try:
        fetch_json(config.EPT_URL)
        print("Endpoint OK.")
    except Exception as e:
        print(f"ERROR: Could not reach EPT endpoint. Check EPT_URL in config.py.\n  {e}")
        sys.exit(1)

    # Parse Study Area Bounds
    clean = config.BOUNDS_STR.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    p = [float(x) for x in clean.split(",")]
    study_area = box(p[0], p[2], p[1], p[3])

    # Generate Overlapping Tiles
    tiles = []
    step = config.TILE_SIZE - config.OVERLAP

    x = p[0]
    while x < p[1]:
        y = p[2]
        while y < p[3]:
            tile = box(x, y, x + config.TILE_SIZE, y + config.TILE_SIZE)
            clipped_tile = tile.intersection(study_area)
            tiles.append(clipped_tile)
            y += step
        x += step

    total = len(tiles)
    mode = "TEST" if config.TEST_RUN else "PRODUCTION"
    print(f"--- {mode} Sync: {total} tiles @ {config.RES}m resolution ---")

    start_time = time.time()
    for i, tile in enumerate(tiles):
        out_path = config.DATA_RAW / f"ground_tile_{i+1:03}.laz"

        if out_path.exists():
            print(f"[{i+1}/{total}] Skipping {out_path.name} (Exists)")
            continue

        tile_start = time.time()
        print(f"[{i+1}/{total}] Downloading {out_path.name}...", end="", flush=True)

        try:
            run_download(tile, str(out_path))
            elapsed = time.time() - tile_start
            remaining = total - (i + 1)
            eta_min = (elapsed * remaining) / 60
            print(f" Done in {elapsed:.1f}s | Est. Remaining: {eta_min:.1f} min")
        except Exception as e:
            print(f" FAILED. Error: {e}")

    print(f"\nTotal Process Complete in {(time.time() - start_time)/60:.2f} minutes.")
