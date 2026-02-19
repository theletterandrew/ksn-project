"""
check_crs.py
------------
Reads the CRS and coordinate info from a single LAS file and reports
everything in the header so we can identify the correct projection.

USAGE:
    1. Set LAS_FILE below.
    2. Run: python check_crs.py
"""

import laspy
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================

LAS_FILE = r"E:\LiDAR\Scoped\Extracted\ground_tile_0.las"

# =============================================================================

def main():
    las_path = Path(LAS_FILE)
    las = laspy.read(str(las_path))

    print("=" * 60)
    print("LAS HEADER INFO")
    print("=" * 60)
    print(f"File version     : {las.header.version}")
    print(f"Point format     : {las.header.point_format.id}")
    print(f"Point count      : {len(las.x):,}")
    print(f"X range          : {las.x.min():.3f} to {las.x.max():.3f}")
    print(f"Y range          : {las.y.min():.3f} to {las.y.max():.3f}")
    print(f"Z range          : {las.z.min():.3f} to {las.z.max():.3f}")
    print(f"X scale / offset : {las.header.scale[0]} / {las.header.offsets[0]}")
    print(f"Y scale / offset : {las.header.scale[1]} / {las.header.offsets[1]}")
    print()

    print("=" * 60)
    print("VLR RECORDS (projection info lives here)")
    print("=" * 60)
    for vlr in las.vlrs:
        print(f"  User ID    : {vlr.user_id!r}")
        print(f"  Record ID  : {vlr.record_id}")
        print(f"  Description: {vlr.description!r}")
        try:
            body = vlr.record_data
            # Try decoding as UTF-8 text (WKT CRS)
            text = body.decode("utf-8", errors="replace").strip()
            if text:
                print(f"  Body (text): {text[:300]}")
        except Exception:
            pass
        print()

    # Also check EVLRs (extra VLRs used in LAS 1.4)
    try:
        if hasattr(las, 'evlrs') and las.evlrs:
            print("=" * 60)
            print("EVLR RECORDS")
            print("=" * 60)
            for evlr in las.evlrs:
                print(f"  User ID    : {evlr.user_id!r}")
                print(f"  Record ID  : {evlr.record_id}")
                try:
                    body = evlr.record_data
                    text = body.decode("utf-8", errors="replace").strip()
                    if text:
                        print(f"  Body (text): {text[:300]}")
                except Exception:
                    pass
                print()
    except Exception:
        pass

if __name__ == "__main__":
    main()
