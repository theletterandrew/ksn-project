import os
import sys
import glob
from pathlib import Path

# Project root (one level up from scripts/)
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

# Calculate the path to the project root (one level up from scripts/)
root_dir = Path(__file__).resolve().parent.parent

# Add the root directory to sys.path so Python can find config.py
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

LAS_FOLDER = config.DATA_PROCESSED

to_delete = [f for f in glob.glob(os.path.join(LAS_FOLDER, "*.las"))
             if os.path.getsize(f) / 1024 < config.MIN_TILE_SIZE_KB]

print(f"Found {len(to_delete)} files to delete:")
for f in to_delete:
    print(f"  {os.path.basename(f)}")

# Confirm before deleting
input("\nPress Enter to delete all of the above, or Ctrl+C to cancel...")

for f in to_delete:
    os.remove(f)
    
print("Done.")