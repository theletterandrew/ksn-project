"""
las_diagnostics.py
------------------
Pre-flight diagnostic for ground-only LiDAR tiles.
Reports density and recommends grid resolution based on actual point counts.
"""

import re
import subprocess
import sys
import logging
from pathlib import Path

# Project root setup
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

# =============================================================================
# CONFIG
# =============================================================================

LAS_DIR      = config.DATA_RAW
LAS_GLOB     = "*.laz"  # Usually downloading LAZ via batchdownload.py
LASINFO_EXE  = config.LASTOOLS_BIN / "lasinfo64.exe"

# Resolution recommendation: target 1.0 ground points per cell for reliability
TARGET_POINTS_PER_CELL = 1.0

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
    """Run lasinfo64.exe and read the report."""
    import tempfile
    tmp = Path(tempfile.mktemp(suffix=".txt"))
    cmd = [str(LASINFO_EXE), "-i", str(las_path), "-cd", "-o", str(tmp)]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        if tmp.exists() and tmp.stat().st_size > 0:
            return tmp.read_text(errors="replace")
        return None
    except Exception as e:
        logger.error(f"  lasinfo64 error for {las_path.name}: {e}")
        return None
    finally:
        tmp.unlink(missing_ok=True)

def parse_lasinfo(output: str, filename: str):
    """Parse lasinfo64 text output into a stats dict."""
    if not output:
        return None

    stats = {"file": filename, "class_breakdown": {}}

    # 1. Capture Total Points from Header
    m = re.search(r"number of point records:\s+([\d,]+)", output, re.IGNORECASE)
    total_points = int(m.group(1).replace(",", "")) if m else 0
    stats["total_points"] = total_points

    # 2. Capture Bounding Box
    m_min = re.search(r"min x y z:\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)", output, re.IGNORECASE)
    m_max = re.search(r"max x y z:\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)", output, re.IGNORECASE)
    
    if m_min and m_max:
        stats["x_min"], stats["y_min"], stats["z_min"] = map(float, m_min.groups())
        stats["x_max"], stats["y_max"], stats["z_max"] = map(float, m_max.groups())
        area_m2 = (stats["x_max"] - stats["x_min"]) * (stats["y_max"] - stats["y_min"])
    else:
        area_m2 = 1.0
    stats["area_m2"] = max(area_m2, 1.0)

    # 3. Handle Ground Points
    # Look for class 2 in the histogram
    ground_count = 0
    for line in output.splitlines():
        m = re.match(r"\s*([\d,]+)\s+\S.*?\(2\)\s*$", line)
        if m:
            ground_count = int(m.group(1).replace(",", ""))
            break
    
    # FALLBACK: If no class 2 found in histogram, but file is large,
    # assume all points are ground points (as per download script logic)
    if ground_count == 0 and total_points > 0:
        ground_count = total_points

    stats["ground_count"] = ground_count
    stats["ground_pct"] = (ground_count / total_points * 100) if total_points > 0 else 0
    stats["density"] = ground_count / stats["area_m2"]
    
    # Calculate Resolution, avoiding ZeroDivision
    if stats["density"] > 0:
        stats["rec_res"] = (TARGET_POINTS_PER_CELL / stats["density"]) ** 0.5
    else:
        stats["rec_res"] = 99.0

    return stats

def main():
    logger = setup_logging()
    files = sorted(Path(LAS_DIR).glob(LAS_GLOB))
    
    if not files:
        logger.error(f"No LAZ files found in {LAS_DIR}")
        sys.exit(1)

    all_stats = []
    logger.info(f"Analyzing {len(files)} ground-only tiles...")
    logger.info("=" * 70)
    logger.info(f"{'File':<25s} | {'Gnd Pts':>10s} | {'Density':>8s} | {'Rec Res':>8s}")
    logger.info("-" * 70)

    for f in files:
        raw = run_lasinfo(f, logger)
        s = parse_lasinfo(raw, f.name)
        if s:
            # Check if tile is empty to avoid local rec_res issues
            if s["ground_count"] == 0:
                logger.warning(f"{s['file']:<25s} | {'0':>10s} | {'0.00':>8s} | {'EMPTY':>8s} !!")
                # We still append to all_stats to keep track of area, 
                # but handle density carefully
            else:
                flag = " !!" if s["rec_res"] > 5.0 else ""
                logger.info(
                    f"{s['file']:<25s} | {s['ground_count']:>10,} | "
                    f"{s['density']:>8.2f} | {s['rec_res']:>6.1f} m{flag}"
                )
            all_stats.append(s)

    # --- Summary Logic Fix ---
    if not all_stats: return
    
    total_pts = sum(s["ground_count"] for s in all_stats)
    total_area = sum(s["area_m2"] for s in all_stats)
    
    # Check for total density to avoid ZeroDivisionError
    avg_density = total_pts / total_area if total_area > 0 else 0
    
    logger.info("=" * 70)
    logger.info(f"SUMMARY")
    logger.info(f"  Total Area        : {total_area/1e6:.2f} km2")
    logger.info(f"  Total Ground Pts  : {total_pts:,}")

    if avg_density > 0:
        avg_res = (TARGET_POINTS_PER_CELL / avg_density) ** 0.5
        logger.info(f"  Overall Density   : {avg_density:.2f} pts/m2")
        logger.info(f"  Suggested Grid    : {avg_res:.1f} m (Config uses {config.RES} m)")
        
        if avg_res > config.RES:
            logger.warning(f"  WARNING: Ground density suggests a coarser grid ({avg_res:.1f}m) "
                           f"than your config.RES ({config.RES}m).")
    else:
        logger.error("  FAILED: No ground points found across any tiles. Check your EPT_URL or BOUNDS.")

if __name__ == "__main__":
    main()