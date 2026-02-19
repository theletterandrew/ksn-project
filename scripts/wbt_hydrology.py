"""
wbt_hydrology.py
----------------
Runs full hydrological conditioning on the mosaicked DEM using
WhiteboxTools. Produces flow direction and flow accumulation rasters
covering the entire study area as a single continuous dataset.

Steps:
    1. Fill depressions (BreachDepressionsLeastCost then FillDepressions)
    2. Flow Direction (D8)
    3. Flow Accumulation (D8)

WhiteboxTools streams data off disk rather than loading entirely into RAM,
making it suitable for large rasters that would overwhelm ArcGIS Pro.

USAGE:
    1. Edit the paths in the CONFIG section below.
    2. Ensure WhiteboxTools is installed and on your PATH:
       whitebox_tools --version
    3. Run from any Python environment (no special packages needed):
       python wbt_hydrology.py

Requirements:
    - WhiteboxTools installed and accessible via PATH
      Download from: https://www.whiteboxgeo.com/download-whiteboxtools/
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

# Calculate the path to the project root (one level up from scripts/)
root_dir = Path(__file__).resolve().parent.parent

# Add the root directory to sys.path so Python can find config.py
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

# =============================================================================
# CONFIG — Edit these before running
# =============================================================================

DEM_MOSAIC  = config.DATA_DEM_MOSAIC   # Input DEM mosaic
OUTPUT_DIR  = config.DATA_SCRATCH_WBT  # Output folder

# WhiteboxTools executable — if on PATH just use "whitebox_tools"
WBT_EXE     = config.WBT_EXE

# =============================================================================
# END CONFIG — No edits needed below this line
# =============================================================================


def setup_logging(output_dir: Path) -> logging.Logger:
    log_path = output_dir / "wbt_hydrology.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def run_wbt(tool: str, args: dict, logger: logging.Logger) -> bool:
    """
    Runs a WhiteboxTools command. Returns True on success.
    args is a dict of parameter name -> value.
    """
    cmd = [WBT_EXE, f"--run={tool}"]
    for key, val in args.items():
        cmd.append(f"--{key}={val}")

    logger.info(f"Running: {tool}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.stdout:
            for line in result.stdout.strip().splitlines():
                logger.info(f"  WBT: {line}")
        if result.stderr:
            for line in result.stderr.strip().splitlines():
                logger.warning(f"  WBT ERR: {line}")

        if result.returncode != 0:
            logger.error(f"{tool} failed with return code {result.returncode}")
            return False

        return True

    except Exception as e:
        logger.error(f"Failed to run {tool}: {e}")
        return False


def main():
    dem_path   = Path(DEM_MOSAIC)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    # Validate input
    if not dem_path.exists():
        logger.error(f"DEM mosaic not found: {dem_path}")
        logger.error("Run mosaic_dem.py first.")
        sys.exit(1)

    # Verify WhiteboxTools is accessible
    try:
        result = subprocess.run(
            [WBT_EXE, "--version"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        logger.info(f"WhiteboxTools version: {result.stdout.strip()}")
    except Exception:
        logger.error("WhiteboxTools not found. Check PATH or WBT_EXE setting.")
        sys.exit(1)

    # Output file paths
    breached_path = output_dir / "dem_breached.tif"
    filled_path   = output_dir / "dem_filled.tif"
    fdr_path      = output_dir / "flow_direction.tif"
    fac_path      = output_dir / "flow_accumulation.tif"

    start_time = time.time()

    # --- Step 1a: Breach depressions (least cost) ---
    # Breaching is preferred over filling for lidar DEMs as it better
    # preserves the natural channel network
    if not breached_path.exists():
        logger.info("=" * 60)
        logger.info("STEP 1a: Breaching depressions (least cost)...")
        logger.info("=" * 60)
        success = run_wbt("BreachDepressionsLeastCost", {
            "dem":      str(dem_path),
            "output":   str(breached_path),
            "dist":     "1000",    # Max breach distance in cells
            "fill":     "true"     # Fill any remaining depressions after breaching
        }, logger)
        if not success:
            logger.error("Breaching failed. Exiting.")
            sys.exit(1)
    else:
        logger.info("STEP 1a: Breached DEM already exists — skipping.")

    # --- Step 1b: Fill remaining depressions ---
    if not filled_path.exists():
        logger.info("=" * 60)
        logger.info("STEP 1b: Filling remaining depressions...")
        logger.info("=" * 60)
        success = run_wbt("FillDepressions", {
            "dem":    str(breached_path),
            "output": str(filled_path),
            "fix_flats": "true"
        }, logger)
        if not success:
            logger.error("Fill failed. Exiting.")
            sys.exit(1)
    else:
        logger.info("STEP 1b: Filled DEM already exists — skipping.")

    # --- Step 2: Flow Direction (D8) ---
    if not fdr_path.exists():
        logger.info("=" * 60)
        logger.info("STEP 2: Computing D8 flow direction...")
        logger.info("=" * 60)
        success = run_wbt("D8Pointer", {
            "dem":    str(filled_path),
            "output": str(fdr_path)
        }, logger)
        if not success:
            logger.error("Flow direction failed. Exiting.")
            sys.exit(1)
    else:
        logger.info("STEP 2: Flow direction already exists — skipping.")

    # --- Step 3: Flow Accumulation (D8) ---
    if not fac_path.exists():
        logger.info("=" * 60)
        logger.info("STEP 3: Computing D8 flow accumulation...")
        logger.info("=" * 60)
        success = run_wbt("D8FlowAccumulation", {
            "input":  str(fdr_path),
            "output": str(fac_path),
            "out_type": "cells"    # Output in number of upstream cells
        }, logger)
        if not success:
            logger.error("Flow accumulation failed. Exiting.")
            sys.exit(1)
    else:
        logger.info("STEP 3: Flow accumulation already exists — skipping.")

    elapsed_total = time.time() - start_time
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info(f"  Breached DEM      : {breached_path}")
    logger.info(f"  Filled DEM        : {filled_path}")
    logger.info(f"  Flow direction    : {fdr_path}")
    logger.info(f"  Flow accumulation : {fac_path}")
    logger.info(f"  Total time        : {elapsed_total / 60:.1f} minutes")
    logger.info("")
    logger.info("Next step: run stream_extraction_wbt.py to extract stream network")


if __name__ == "__main__":
    main()
