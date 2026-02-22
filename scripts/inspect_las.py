"""
inspect_las.py
--------------
Prints summary statistics for all LAS files in the processed data directory.
"""

import sys
import laspy
import numpy as np
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

las_files = sorted(config.DATA_PROCESSED.glob("*.las"))

if not las_files:
    print(f"No LAS files found in: {config.DATA_PROCESSED}")
    sys.exit(1)

print(f"Found {len(las_files)} LAS files in {config.DATA_PROCESSED}")
print("=" * 70)

total_points = 0

for las_path in las_files:
    with laspy.open(las_path) as reader:
        las = reader.read()

    arr = las.points.array
    x_coords = las.header.offsets[0] + las.header.scales[0] * arr["X"].astype(np.float64)
    y_coords = las.header.offsets[1] + las.header.scales[1] * arr["Y"].astype(np.float64)
    z_coords = las.header.offsets[2] + las.header.scales[2] * arr["Z"].astype(np.float64)

    classifications = arr["raw_classification"]
    unique_classes, class_counts = np.unique(classifications, return_counts=True)

    size_mb = las_path.stat().st_size / (1024 * 1024)
    n_points = len(arr)
    total_points += n_points

    print(f"File:            {las_path.name}  ({size_mb:.1f} MB)")
    print(f"Points:          {n_points:,}")
    print(f"X range:         {x_coords.min():.1f}  to  {x_coords.max():.1f}")
    print(f"Y range:         {y_coords.min():.1f}  to  {y_coords.max():.1f}")
    print(f"Z range:         {z_coords.min():.1f}  to  {z_coords.max():.1f}  (meters, NAVD88)")
    print(f"Classifications: ", end="")
    print(", ".join(f"class {c}: {n:,}" for c, n in zip(unique_classes, class_counts)))
    print("-" * 70)

print(f"Total points across all files: {total_points:,}")
