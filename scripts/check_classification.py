"""
check_classification.py
-----------------------
Reads a single LAS file and reports what classification codes
are present in the data. Run this before batch processing to
confirm the correct class filter to use in las_to_dem.py.

USAGE:
    1. Set LAS_FILE to any one of your LAS files below.
    2. Run: python check_classification.py
"""

from pathlib import Path
import json
import pdal
import numpy as np

# =============================================================================
# CONFIG — point this at any single LAS file from your dataset
# =============================================================================

LAS_FILE = r"E:\LiDAR\Scoped\Extracted\ground_tile_0.las"

# =============================================================================
# END CONFIG
# =============================================================================


def main():
    las_path = Path(LAS_FILE)

    if not las_path.exists():
        print(f"ERROR: File not found: {las_path}")
        return

    print(f"Reading: {las_path.name}")
    print("-" * 50)

    # Read the LAS file into a numpy array via PDAL
    pipeline = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": str(las_path)
            }
        ]
    }

    p = pdal.Pipeline(json.dumps(pipeline))
    p.execute()

    # Get the point data as a numpy array
    arrays = p.arrays
    if not arrays or len(arrays[0]) == 0:
        print("ERROR: No points were read from the file.")
        return

    points = arrays[0]
    total_points = len(points)
    print(f"Total points in file : {total_points:,}")
    print()

    # Check what fields are available
    print(f"Available fields     : {list(points.dtype.names)}")
    print()

    # Report classification breakdown
    if "Classification" in points.dtype.names:
        classifications = points["Classification"]
        unique, counts = np.unique(classifications, return_counts=True)

        print("Classification breakdown:")
        for cls, count in zip(unique, counts):
            pct = count / total_points * 100
            print(f"  Class {cls:3d} : {count:>10,} points  ({pct:.1f}%)")
    else:
        print("WARNING: No 'Classification' field found in this LAS file.")
        print("The file may not have classification data at all.")

    # Also report Z range as a sanity check
    if "Z" in points.dtype.names:
        z = points["Z"]
        print()
        print(f"Z (elevation) range  : {z.min():.2f} to {z.max():.2f} meters")
    else:
        print("WARNING: No 'Z' field found — file may be malformed.")


if __name__ == "__main__":
    main()
