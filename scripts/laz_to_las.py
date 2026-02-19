"""
laz_to_las.py
-------------
Batch extracts LAZ files to LAS format using LASzip.
Outputs all LAS files to E:\LiDAR\Scoped\Extracted, creating the folder if needed.

USAGE:
    1. Edit the paths and settings in the CONFIG section below.
    2. Run: python laz_to_las.py

Requirements:
    LASzip must be installed and laszip.exe must be accessible.
    Download from: https://laszip.org
"""

import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Calculate the path to the project root (one level up from scripts/)
root_dir = Path(__file__).resolve().parent.parent

# Add the root directory to sys.path so Python can find config.py
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

# CONFIG 


INPUT_DIR   = config.DATA_RAW      # Folder containing .laz files
OUTPUT_DIR  = config.DATA_PROCESSED       # Output folder (created if missing)

LASZIP_EXE  = config.LASZIP_EXE  # Path to laszip.exe

MAX_WORKERS = config.MAX_WORKERS        # Number of parallel extractions



def setup_logging(output_dir: Path) -> logging.Logger:
    log_path = output_dir / "laz_to_las.log"
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


def extract_tile(laz_path: Path, output_dir: Path, laszip_exe: str) -> tuple[str, bool, str]:
    """
    Extracts a single LAZ file to LAS using laszip.exe.
    Returns (filename, success, message).
    """
    out_path = output_dir / (laz_path.stem + ".las")

    # Skip if already extracted — safe to re-run after interruptions
    if out_path.exists():
        return (laz_path.name, True, "Skipped — already exists")

    try:
        cmd = [
            laszip_exe,
            "-i", str(laz_path),
            "-o", str(out_path)
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            return (laz_path.name, False, error_msg)

        if not out_path.exists() or out_path.stat().st_size == 0:
            return (laz_path.name, False, "Output file missing or empty after extraction")

        return (laz_path.name, True, f"OK -> {out_path.name}")

    except Exception as e:
        if out_path.exists():
            out_path.unlink()  # Remove partial output
        return (laz_path.name, False, str(e))


def main():
    input_dir  = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    # Verify laszip.exe exists
    if not Path(LASZIP_EXE).exists():
        logger.error(f"laszip.exe not found at: {LASZIP_EXE}")
        logger.error("Download LASzip from https://laszip.org and update LASZIP_EXE in the config.")
        sys.exit(1)

    # Collect all LAZ files
    laz_files = sorted(input_dir.glob("*.laz"))
    if not laz_files:
        logger.error(f"No .laz files found in: {input_dir}")
        sys.exit(1)

    total = len(laz_files)
    logger.info(f"Found {total} LAZ files")
    logger.info(f"Input dir  : {input_dir}")
    logger.info(f"Output dir : {output_dir}")
    logger.info(f"LASzip exe : {LASZIP_EXE}")
    logger.info(f"Workers    : {MAX_WORKERS}")
    logger.info("-" * 60)

    succeeded, failed, skipped = 0, 0, 0
    failures = []
    start_time = time.time()

    # ThreadPoolExecutor is used here (vs ProcessPoolExecutor) because
    # each worker is spawning an external subprocess, not running Python code
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(extract_tile, laz_path, output_dir, LASZIP_EXE): laz_path
            for laz_path in laz_files
        }

        for i, future in enumerate(as_completed(futures), start=1):
            filename, success, message = future.result()

            if "Skipped" in message:
                skipped += 1
                status = "SKIP"
            elif success:
                succeeded += 1
                status = "OK  "
            else:
                failed += 1
                failures.append((filename, message))
                status = "FAIL"

            elapsed = time.time() - start_time
            rate    = i / elapsed
            eta_min = (total - i) / rate / 60 if rate > 0 else 0

            logger.info(
                f"[{i:4d}/{total}] {status}  {filename}  |  "
                f"{elapsed/60:.1f} min elapsed  |  ETA {eta_min:.1f} min  |  {message}"
            )

    # Final summary
    elapsed_total = time.time() - start_time
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info(f"  Succeeded : {succeeded}")
    logger.info(f"  Skipped   : {skipped}")
    logger.info(f"  Failed    : {failed}")
    logger.info(f"  Total time: {elapsed_total / 60:.1f} minutes")

    if failures:
        logger.error("Failed tiles:")
        for fname, msg in failures:
            logger.error(f"  {fname}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
