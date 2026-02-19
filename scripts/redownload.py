import os
import glob
import numpy as np
import pdal
import json

LAS_FOLDER = r"E:\LiDAR\Scoped\ground_tiles"
EPT_URL = "http://usgs-lidar-public.s3.amazonaws.com/USGS_LPC_CA_SoCal_Wildfires_B1_2018_LAS_2019/ept.json"
TILE_SIZE = 5000
RES = 2.0

XMIN = -13035749.581531966
YMIN =  4018953.87470956
STEP = 3000
N_COLS = 21

# Find all empty tiles
empty_tiles = []
for f in sorted(glob.glob(os.path.join(LAS_FOLDER, "*.laz")),
                key=lambda p: int(os.path.basename(p).replace("ground_tile_","").replace(".laz",""))):
    if os.path.getsize(f) / 1024 < 10:
        empty_tiles.append(f)

print(f"Re-downloading {len(empty_tiles)} empty tiles...")

for laz_path in empty_tiles:
    idx   = int(os.path.basename(laz_path).replace("ground_tile_","").replace(".laz",""))
    col   = idx % N_COLS
    row   = idx // N_COLS
    x0    = XMIN + col * STEP
    y0    = YMIN + row * STEP

    print(f"  [{idx}] col={col} row={row}", end="", flush=True)

    pipeline_dict = {
        "pipeline": [
            {
                "type": "readers.ept",
                "filename": EPT_URL,
                "bounds": f"([{x0}, {x0 + TILE_SIZE}], [{y0}, {y0 + TILE_SIZE}])",
                "resolution": RES
            },
            {"type": "filters.range", "limits": "Classification[2:2]"},
            {"type": "writers.las", "filename": laz_path, "compression": "laszip"}
        ]
    }

    try:
        pipeline = pdal.Pipeline(json.dumps(pipeline_dict))
        pipeline.execute()
        new_size = os.path.getsize(laz_path) / 1024
        print(f" → {new_size:.1f} kb")
    except Exception as e:
        print(f" FAILED: {e}")