import pdal
import sys
from pathlib import Path
import json
from shapely.geometry import box
import time

# Calculate the path to the project root (one level up from scripts/)
root_dir = Path(__file__).resolve().parent.parent

# Add the root directory to sys.path so Python can find config.py
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

def run_download(tile_box, filename):
    """
    Executes PDAL pipeline to fetch EPT data for a specific tile.
    Note: PDAL EPT bounds format is ([minx, maxx], [miny, maxy])
    """
    b = tile_box.bounds # returns (minx, miny, maxx, maxy)
    
    pipeline_dict = {
        "pipeline": [
            {
                "type": "readers.ept",
                "filename": config.EPT_URL,
                # PDAL EPT bounds: ([xmin, xmax], [ymin, ymax])
                "bounds": f"([{b[0]}, {b[2]}], [{b[1]}, {b[3]}])",
                "resolution": config.RES 
            },
            { 
                "type": "filters.range", 
                "limits": "Classification[2:2]" # 2 is Ground in LAS spec
            },
            { 
                "type": "writers.las", 
                "filename": filename, 
                "compression": "laszip" 
            }
        ]
    }
    pipeline = pdal.Pipeline(json.dumps(pipeline_dict))
    pipeline.execute()

if __name__ == "__main__":
    # Ensure output directory exists
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)

    # Parse Study Area Bounds
    clean = config.BOUNDS_STR.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    p = [float(x) for x in clean.split(',')]
    study_area = box(p[0], p[2], p[1], p[3])

    # Generate Overlapping Tiles
    tiles = []
    step = config.TILE_SIZE - config.TILE_OVERLAP

    # Use while loops to ensure we cover the whole area
    x = p[0]
    while x < p[1]:
        y = p[2]
        while y < p[3]:
            # Create tile and clip it to the study area boundary
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
        # We use i+1 for the filename so it's 1-indexed for the user
        out_path = config.DATA_RAW / f"ground_tile_{i+1:03}.laz"
        
        if out_path.exists():
            print(f"[{i+1}/{total}] Skipping {out_path.name} (Exists)")
            continue

        tile_start = time.time()
        print(f"[{i+1}/{total}] Downloading {out_path.name}...", end="", flush=True)

        tile_start = time.time()
        print(f"[{i+1}/{total}] Downloading {out_path.name}...", end="", flush=True)
        
        try:
            run_download(tile, str(out_path))
            elapsed = time.time() - tile_start
            
            # ETA Calculation
            remaining = total - (i + 1)
            eta_min = (elapsed * remaining) / 60
            
            print(f" Done in {elapsed:.1f}s | Est. Remaining: {eta_min:.1f} min")
        except Exception as e:
            print(f" FAILED. Error: {e}")

    print(f"\nTotal Process Complete in {(time.time() - start_time)/60:.2f} minutes.")