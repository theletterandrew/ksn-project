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
    """Parse lasinfo output into essential ground-only stats."""
    if not output:
        return None

    stats = {"file": filename}

    # Total points (which are all ground points in this workflow)
    m = re.search(r"number of point records:\s+([\d,]+)", output, re.IGNORECASE)
    total_pts = int(m.group(1).replace(",", "")) if m else 0
    stats["ground_count"] = total_pts

    # Bounding box & Area
    m_min = re.search(r"min x y z:\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)", output, re.IGNORECASE)
    m_max = re.search(r"max x y z:\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)", output, re.IGNORECASE)
    
    if m_min and m_max:
        area_m2 = (float(m_max.group(1)) - float(m_min.group(1))) * \
                  (float(m_max.group(2)) - float(m_min.group(2)))
        stats["area_m2"] = max(area_m2, 1.0)
        stats["z_range"] = (float(m_min.group(3)), float(m_max.group(3)))
    else:
        stats["area_m2"] = 1.0

    # Density & Resolution
    stats["density"] = stats["ground_count"] / stats["area_m2"]
    stats["rec_res"] = (TARGET_POINTS_PER_CELL / stats["density"]) ** 0.5 if stats["density"] > 0 else 99.0
    
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
            flag = " !!" if s["rec_res"] > 5.0 else ""
            logger.info(
                f"{s['file']:<25s} | {s['ground_count']:>10,} | "
                f"{s['density']:>8.2f} | {s['rec_res']:>6.1f} m{flag}"
            )
            all_stats.append(s)

    # Summary
    if not all_stats: return
    
    total_pts = sum(s["ground_count"] for s in all_stats)
    total_area = sum(s["area_m2"] for s in all_stats)
    avg_density = total_pts / total_area
    avg_res = (TARGET_POINTS_PER_CELL / avg_density) ** 0.5

    logger.info("=" * 70)
    logger.info(f"SUMMARY")
    logger.info(f"  Total Area        : {total_area/1e6:.2f} km2")
    logger.info(f"  Total Ground Pts  : {total_pts:,}")
    logger.info(f"  Overall Density   : {avg_density:.2f} pts/m2")
    logger.info(f"  Suggested Grid    : {avg_res:.1f} m (Config uses {config.RES} m)")
    
    if avg_res > config.RES:
        logger.warning(f"  WARNING: Ground density suggests a coarser grid ({avg_res:.1f}m) "
                       f"than your config.RES ({config.RES}m).")

if __name__ == "__main__":
    main()