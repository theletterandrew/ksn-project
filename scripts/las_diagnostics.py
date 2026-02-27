"""
las_diagnostics.py
------------------
Pre-flight diagnostic for a folder of LAS/LAZ files using LAStools
lasinfo64.exe. Reports per-tile and summary statistics on:
  - Total points and classification breakdown
  - Ground point (class 2) count and percentage
  - Bounding box and tile area
  - Point density (pts/m2) for all points and ground-only
  - Recommended raster resolution based on ground density
  - Flags tiles with suspiciously low ground returns

Run this BEFORE LAS_to_dem.py to understand your data and choose
an appropriate grid resolution.

USAGE:
    python las_diagnostics.py

Requirements:
    - LAStools lasinfo64.exe (configured via config.LASTOOLS_BIN)
    - Completed download/extraction steps first
"""

import re
import subprocess
import sys
import logging
from pathlib import Path

# =============================================================================
# Project root setup
# =============================================================================
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

# =============================================================================
# CONFIG
# =============================================================================

LAS_DIR      = config.DATA_RAW
LAS_GLOB     = "*.las"
LASINFO_EXE  = config.LASTOOLS_BIN / "lasinfo64.exe"

# Resolution recommendation: aim for this many ground points per output cell.
# 1.0 = at least 1 ground point per cell (minimum reliable)
# 4.0 = 4 ground points per cell (good quality)
TARGET_POINTS_PER_CELL = 1.0

CLASS_NAMES = {
    0:  "Never classified",
    1:  "Unclassified",
    2:  "Ground",
    3:  "Low vegetation",
    4:  "Medium vegetation",
    5:  "High vegetation",
    6:  "Building",
    7:  "Low point (noise)",
    8:  "Reserved",
    9:  "Water",
    10: "Rail",
    11: "Road surface",
    17: "Bridge deck",
    18: "High noise",
}

# =============================================================================
# END CONFIG
# =============================================================================


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def run_lasinfo(las_path: Path, logger: logging.Logger):
    """Run lasinfo64.exe on a file and return its stdout as a string."""
    cmd = [str(LASINFO_EXE), "-i", str(las_path), "-cd", "-o", "stdout"]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.error(f"  lasinfo64 failed for {las_path.name}: {result.stderr.strip()}")
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        logger.error(f"  lasinfo64 timed out for {las_path.name}")
        return None
    except FileNotFoundError:
        logger.error(f"lasinfo64.exe not found at {LASINFO_EXE}. Check config.LASTOOLS_BIN.")
        sys.exit(1)


def parse_lasinfo(output: str, filename: str):
    """Parse lasinfo64 text output into a stats dict."""
    if not output:
        return None

    stats = {"file": filename, "class_breakdown": {}}

    # Total points
    m = re.search(r"number of point records:\s+([\d,]+)", output, re.IGNORECASE)
    if m:
        stats["total_points"] = int(m.group(1).replace(",", ""))
    else:
        m = re.search(r"reporting all ([\d,]+) points", output, re.IGNORECASE)
        stats["total_points"] = int(m.group(1).replace(",", "")) if m else 0

    # Bounding box
    m = re.search(r"min x y z:\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)", output, re.IGNORECASE)
    if m:
        stats["x_min"], stats["y_min"], stats["z_min"] = float(m.group(1)), float(m.group(2)), float(m.group(3))
    m = re.search(r"max x y z:\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)", output, re.IGNORECASE)
    if m:
        stats["x_max"], stats["y_max"], stats["z_max"] = float(m.group(1)), float(m.group(2)), float(m.group(3))

    # Classification histogram block
    # lasinfo prints lines like:   4561234  ground (2)
    class_block = re.search(
        r"classification histogram.*?(?=\n\s*\n|\Z)", output, re.IGNORECASE | re.DOTALL
    )
    if class_block:
        for line in class_block.group(0).splitlines()[1:]:
            m = re.match(r"\s*([\d,]+)\s+\S.*?\((\d+)\)", line)
            if m:
                code  = int(m.group(2))
                count = int(m.group(1).replace(",", ""))
                stats["class_breakdown"][code] = stats["class_breakdown"].get(code, 0) + count

    # Derived stats
    x_range = stats.get("x_max", 0) - stats.get("x_min", 0)
    y_range = stats.get("y_max", 0) - stats.get("y_min", 0)
    area_m2 = x_range * y_range if x_range > 0 and y_range > 0 else 1.0
    stats["area_m2"] = area_m2

    total  = stats.get("total_points", 0)
    ground = stats["class_breakdown"].get(2, 0)
    stats["ground_count"]   = ground
    stats["ground_pct"]     = 100.0 * ground / total if total > 0 else 0.0
    stats["all_density"]    = total  / area_m2
    stats["ground_density"] = ground / area_m2

    def rec_res(density):
        return (TARGET_POINTS_PER_CELL / density) ** 0.5 if density > 0 else float("inf")

    stats["rec_res_all"]    = rec_res(stats["all_density"])
    stats["rec_res_ground"] = rec_res(stats["ground_density"])
    return stats


def print_tile_report(s: dict, logger: logging.Logger):
    z_str = f"{s.get('z_min',0):.1f} - {s.get('z_max',0):.1f} m" if "z_min" in s else "unknown"
    logger.info(f"  File        : {s['file']}")
    logger.info(f"  Total pts   : {s.get('total_points', 0):,}")
    logger.info(f"  Area        : {s['area_m2']/1e6:.3f} km2")
    logger.info(f"  Z range     : {z_str}")
    logger.info(f"  All density : {s['all_density']:.4f} pts/m2  (rec. res ~{s['rec_res_all']:.1f} m)")
    logger.info(f"  Ground pts  : {s['ground_count']:,}  ({s['ground_pct']:.1f}%)")
    logger.info(f"  Gnd density : {s['ground_density']:.4f} pts/m2  (rec. res ~{s['rec_res_ground']:.1f} m)")

    logger.info("  Classifications:")
    total = s.get("total_points", 1)
    for code, count in sorted(s["class_breakdown"].items()):
        name = CLASS_NAMES.get(code, f"Class {code}")
        pct  = 100.0 * count / total
        logger.info(f"    [{code:2d}] {name:<25s}  {count:>10,}  ({pct:5.1f}%)")

    if not s["class_breakdown"]:
        logger.warning("  !! No classification histogram found in lasinfo output")
    if s["ground_pct"] < 5.0 and s.get("total_points", 0) > 0:
        logger.warning(f"  !! LOW GROUND RETURN: only {s['ground_pct']:.1f}% ground points")
    if s["ground_count"] == 0:
        logger.warning("  !! NO GROUND POINTS -- tile may not be classified")
    if s["rec_res_ground"] > 5.0:
        logger.warning(
            f"  !! Rec. ground resolution ({s['rec_res_ground']:.1f} m) is coarser than 5 m "
            f"-- gridding at 2 m will require heavy gap-filling"
        )


def main():
    logger = setup_logging()
    las_dir = Path(LAS_DIR)

    if not LASINFO_EXE.exists():
        logger.error(f"lasinfo64.exe not found at {LASINFO_EXE}")
        logger.error("Check config.LASTOOLS_BIN is pointing at the right folder.")
        sys.exit(1)

    files = sorted(las_dir.glob(LAS_GLOB))
    if not files:
        files = sorted(las_dir.glob("*.laz"))
    if not files:
        logger.error(f"No LAS/LAZ files found in {las_dir}")
        sys.exit(1)

    logger.info(f"lasinfo64 : {LASINFO_EXE}")
    logger.info(f"Input dir : {las_dir}")
    logger.info(f"Files     : {len(files)}")
    logger.info("=" * 60)

    all_stats = []
    for i, f in enumerate(files, 1):
        logger.info(f"[{i:3d}/{len(files)}] {f.name}")
        raw   = run_lasinfo(f, logger)
        stats = parse_lasinfo(raw, f.name) if raw else None
        if stats:
            print_tile_report(stats, logger)
            all_stats.append(stats)
        else:
            logger.warning(f"  Could not parse stats for {f.name}")
        logger.info("-" * 60)

    if not all_stats:
        logger.error("No tiles could be read.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    total_pts    = sum(s.get("total_points", 0) for s in all_stats)
    total_area   = sum(s["area_m2"] for s in all_stats)
    total_ground = sum(s["ground_count"] for s in all_stats)

    def rec_res(density):
        return (TARGET_POINTS_PER_CELL / density) ** 0.5 if density > 0 else float("inf")

    overall_all_density    = total_pts    / total_area if total_area > 0 else 0
    overall_ground_density = total_ground / total_area if total_area > 0 else 0

    logger.info(f"  Tiles processed     : {len(all_stats)}")
    logger.info(f"  Total points        : {total_pts:,}")
    logger.info(f"  Total area          : {total_area/1e6:.2f} km2")
    if total_pts:
        logger.info(f"  Total ground pts    : {total_ground:,}  ({100*total_ground/total_pts:.1f}%)")
    logger.info(f"  Overall all density : {overall_all_density:.4f} pts/m2")
    logger.info(f"  Overall gnd density : {overall_ground_density:.4f} pts/m2")
    logger.info(f"  Rec. res (all pts)  : ~{rec_res(overall_all_density):.1f} m")
    logger.info(f"  Rec. res (gnd only) : ~{rec_res(overall_ground_density):.1f} m")
    logger.info("")

    # Per-tile table
    logger.info(f"  {'File':<25s}  {'Gnd pts':>10s}  {'Gnd%':>6s}  {'Density':>10s}  {'Rec res':>8s}")
    logger.info("  " + "-" * 67)
    for s in all_stats:
        flag = " !!" if s["rec_res_ground"] > 5.0 or s["ground_count"] == 0 else ""
        logger.info(
            f"  {s['file']:<25s}  {s['ground_count']:>10,}  "
            f"{s['ground_pct']:>5.1f}%  "
            f"{s['ground_density']:>10.4f}  "
            f"{s['rec_res_ground']:>7.1f} m{flag}"
        )

    rec = rec_res(overall_ground_density)
    logger.info("")
    logger.info(f"RECOMMENDATION: Grid at ~{rec:.0f} m resolution based on overall ground point density.")
    if rec > 5.0:
        logger.warning(
            "Ground density is low. Consider using USGS 3DEP 10m data as a base layer, "
            "or lower STREAM_THRESHOLD to match resolution."
        )


if __name__ == "__main__":
    main()
