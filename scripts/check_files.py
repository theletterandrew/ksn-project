import os
import glob

LAS_FOLDER = r"E:\LiDAR\Scoped\ground_tiles"

sizes = {f: os.path.getsize(f)/1024 
         for f in glob.glob(os.path.join(LAS_FOLDER, "*.laz"))}

empty  = [f for f, kb in sizes.items() if kb < 10]
filled = [f for f, kb in sizes.items() if kb >= 10]

print(f"Valid tiles:  {len(filled)}")
print(f"Empty tiles:  {len(empty)}")