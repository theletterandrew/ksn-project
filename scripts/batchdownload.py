"""
batchdownload.py
----------------
Downloads ground-classified LiDAR point cloud tiles from a USGS EPT endpoint
using the EPT REST API (requests) and filters to ground points (Classification=2)
using laspy. Outputs LAZ-compressed files identical to the original PDAL pipeline.

No PDAL dependency required.
"""

import sys
import json
import time
import requests
import laspy
from pathlib import Path
from shapely.geometry import box

# Calculate the path to the project root (one level up from scripts/)
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config


def fetch_ept_info(ept_url: str) -> dict:
    """Fetch the EPT metadata JSON to verify the endpoint is reachable."""
    response = requests.get(ept_url, timeout=30)
    response.raise_for_status()
    return response.json()


def run_download(tile_box, filename: str):
    """
    Downloads a spatial subset of an EPT dataset for a given tile bounding box,
    filters to ground-only points (Classification=2), and writes a LAZ file.

    EPT tiles are fetched as binary LAS via the EPT REST API. Filtering and
    LAZ writing are handled by laspy, which is already in ksn_env.
    """
    b = tile_box.bounds  # (minx, miny, maxx, maxy)
    ept_base = config.EPT_URL.rsplit("/", 1)[0]  # strip ept.json from URL

    # --- Step 1: Request point data from the EPT REST endpoint ---
    # The EPT /read endpoint accepts a bounds query and returns binary LAS
    params = {
        "bounds": json.dumps([b[0], b[1], b[2], b[3]]),  # [minx, miny, maxx, maxy]
        "resolution": config.RES
    }
    read_url = f"{ept_base}/read"
    response = requests.get(read_url, params=params, timeout=300, stream=True)
    response.raise_for_status()

    # --- Step 2: Write raw response to a temp LAS file ---
    tmp_path = Path(filename).with_suffix(".tmp.las")
    try:
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # --- Step 3: Filter to ground points only (Classification=2) ---
        with laspy.open(tmp_path) as reader:
            las = reader.read()

        ground_mask = las.classification == 2
        ground_points = las.points[ground_mask]

        if len(ground_points) == 0:
            raise ValueError(
                "No ground points found in this tile — "
                "tile may be empty, offshore, or outside the dataset extent."
            )

        # --- Step 4: Write filtered points as LAZ ---
        filtered = laspy.LasData(header=las.header)
        filtered.points = ground_points

        with laspy.open(
            filename,
            mode="w",
            header=filtered.header,
            laz_backend=laspy.LazBackend.LazrsParallel
        ) as writer:
            writer.write_points(filtered.points)

    finally:
        # Always clean up the temp file, even if something fails
        if tmp_path.exists():
            tmp_path.unlink()


if __name__ == "__main__":
    # Ensure output directory exists
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)

    # Verify EPT endpoint is reachable before starting the loop
    print(f"Verifying EPT endpoint: {config.EPT_URL}")
    try:
        fetch_ept_info(config.EPT_URL)
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

    # Process Loop
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
