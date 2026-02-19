"""
check_laz_crs.py
----------------
Reads and reports all CRS information (horizontal and vertical) from a
LAZ or LAS file header, including WKT strings, EPSG codes, and geoid model.

USAGE:
    1. Set LAZ_FILE below to any one of your LAZ or LAS files.
    2. Run: python check_laz_crs.py

Requirements:
    conda install -c conda-forge laspy lazrs-python pyproj
"""

import laspy
from pathlib import Path
from pyproj import CRS

# =============================================================================
# CONFIG
# =============================================================================

LAZ_FILE = r"E:\LiDAR\Scoped\ground_tiles\ground_tile_0.laz"   # Point at any LAZ or LAS file

# =============================================================================
# END CONFIG
# =============================================================================


def try_parse_wkt(wkt: str) -> str:
    """Attempt to parse a WKT string and return a human-readable summary."""
    try:
        crs = CRS.from_wkt(wkt)
        lines = []
        lines.append(f"    Name       : {crs.name}")
        lines.append(f"    Type       : {crs.type_name}")
        try:
            auth = crs.to_authority()
            if auth:
                lines.append(f"    EPSG/Auth  : {auth[0]}:{auth[1]}")
        except Exception:
            pass
        # Check for compound CRS (horizontal + vertical combined)
        if crs.is_compound:
            lines.append(f"    ** COMPOUND CRS — contains both horizontal and vertical **")
            try:
                sub = crs.sub_crs_list
                for i, s in enumerate(sub):
                    lines.append(f"    Sub CRS {i+1}  : {s.name} ({s.type_name})")
                    try:
                        auth = s.to_authority()
                        if auth:
                            lines.append(f"               EPSG: {auth[0]}:{auth[1]}")
                    except Exception:
                        pass
            except Exception:
                pass
        return "\n".join(lines)
    except Exception as e:
        return f"    (Could not parse WKT: {e})"


def main():
    path = Path(LAZ_FILE)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return

    print(f"File: {path.name}")
    las = laspy.read(str(path))

    print("=" * 60)
    print("COORDINATE RANGES")
    print("=" * 60)
    print(f"  X : {las.x.min():.3f} to {las.x.max():.3f}")
    print(f"  Y : {las.y.min():.3f} to {las.y.max():.3f}")
    print(f"  Z : {las.z.min():.3f} to {las.z.max():.3f}")
    print()

    print("=" * 60)
    print("VLR RECORDS — Full CRS Dump")
    print("=" * 60)

    found_any = False
    for vlr in las.vlrs:
        user_id   = vlr.user_id.strip()
        record_id = vlr.record_id

        # Only print VLRs that are likely to contain CRS info
        is_projection = (
            user_id == "LASF_Projection" or
            user_id == "liblas" or
            "proj" in user_id.lower() or
            "crs" in user_id.lower() or
            "srs" in user_id.lower() or
            record_id in (2111, 2112, 34735, 34736, 34737)
        )

        if not is_projection:
            continue

        found_any = True
        print(f"  User ID    : {user_id!r}")
        print(f"  Record ID  : {record_id}")
        print(f"  Description: {vlr.description!r}")

        # Record ID 2112 = OGC WKT CRS (most informative)
        if record_id == 2112:
            try:
                wkt = vlr.record_data.decode("utf-8", errors="replace").strip().rstrip("\x00")
                print(f"  WKT string : {wkt[:500]}")
                print()
                print("  Parsed CRS info:")
                print(try_parse_wkt(wkt))
            except Exception as e:
                print(f"  (Could not decode WKT: {e})")

        # Record ID 2111 = GeoAsciiParamsTag (short text description)
        elif record_id == 2111:
            try:
                text = vlr.record_data.decode("utf-8", errors="replace").strip().rstrip("\x00")
                print(f"  GeoAscii   : {text!r}")
            except Exception as e:
                print(f"  (Could not decode: {e})")

        # Record ID 34735 = GeoKeyDirectoryTag
        elif record_id == 34735:
            print("  GeoKeyDirectoryTag present (older GeoTIFF-style CRS encoding)")

        print()

    # Also check EVLRs (LAS 1.4 extended VLRs — vertical info sometimes lives here)
    print("=" * 60)
    print("EVLR RECORDS (LAS 1.4 extended VLRs)")
    print("=" * 60)
    try:
        if hasattr(las, 'evlrs') and las.evlrs:
            for evlr in las.evlrs:
                print(f"  User ID    : {evlr.user_id!r}")
                print(f"  Record ID  : {evlr.record_id}")
                try:
                    wkt = evlr.record_data.decode("utf-8", errors="replace").strip().rstrip("\x00")
                    if wkt:
                        print(f"  WKT string : {wkt[:500]}")
                        print()
                        print("  Parsed CRS info:")
                        print(try_parse_wkt(wkt))
                except Exception as e:
                    print(f"  (Could not decode: {e})")
                print()
        else:
            print("  No EVLRs found.")
    except Exception as e:
        print(f"  Error reading EVLRs: {e}")

    if not found_any:
        print()
        print("WARNING: No projection VLRs found. File may have no embedded CRS.")


if __name__ == "__main__":
    main()
