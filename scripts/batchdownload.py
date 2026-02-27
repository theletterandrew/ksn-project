import pdal
import json
import time
import sys
from pathlib import Path
from shapely.geometry import box

# Project root setup
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

def run_download_pdal(minx, maxx, miny, maxy, filename: str):
    """
    Executes a PDAL pipeline for a specific tile.
    """
    # PDAL bounds format: ([minx, maxx], [miny, maxy])
    pdal_bounds = f"([{minx}, {maxx}], [{miny}, {maxy}])"

    pipeline_definition = [
        {
            "type": "readers.ept",
            "filename": config.EPT_URL,
            "bounds": pdal_bounds,
            "resolution": 0.01  # Native resolution
        },
        {
            "type": "filters.range",
            "limits": "Classification[2:2]" # Ground only
        },
        {
            "type": "writers.las",
            "filename": filename,
            "compression": "lazperf"
        }
    ]

    pipeline = pdal.Pipeline(json.dumps(pipeline_definition))
    pipeline.execute()

if __name__ == "__main__":
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)

    # 1. Parse Study Area Bounds from config.py
    # Re-using your logic to strip brackets/parentheses
    clean = config.BOUNDS_STR.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    p = [float(x) for x in clean.split(",")]
    
    # Coordinates are expected as: minx, maxx, miny, maxy
    study_area = box(p[0], p[2], p[1], p[3])

    # 2. Generate Overlapping Tiles
    tiles = []
    step = config.TILE_SIZE - config.OVERLAP

    x = p[0]
    while x < p[1]:
        y = p[2]
        while y < p[3]:
            # Create tile and clip to study area to avoid requesting empty space
            tile_geom = box(x, y, x + config.TILE_SIZE, y + config.TILE_SIZE)
            clipped_tile = tile_geom.intersection(study_area)
            if not clipped_tile.is_empty:
                tiles.append(clipped_tile.bounds) # (minx, miny, maxx, maxy)
            y += step
        x += step

    total = len(tiles)
    mode = "TEST" if config.TEST_RUN else "PRODUCTION"
    print(f"--- {mode} PDAL Sync: {total} tiles ---")

    # 3. Processing Loop
    start_time = time.time()
    for i, (t_minx, t_miny, t_maxx, t_maxy) in enumerate(tiles):
        out_path = config.DATA_RAW / f"gt_{i+1:03}.laz"

        if out_path.exists():
            print(f"[{i+1}/{total}] Skipping {out_path.name} (Exists)")
            continue

        print(f"[{i+1}/{total}] Downloading {out_path.name}...", end="", flush=True)
        tile_start = time.time()

        try:
            # Pass individual tile bounds to the PDAL pipeline
            run_download_pdal(t_minx, t_maxx, t_miny, t_maxy, str(out_path))
            
            elapsed = time.time() - tile_start
            print(f" Done in {elapsed:.1f}s")
        except Exception as e:
            print(f" FAILED: {e}")

    print(f"\nProcess Complete in {(time.time() - start_time)/60:.2f} minutes.")