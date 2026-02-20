import subprocess
import sys
import time
import os

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")

SCRIPTS_TO_RUN = [
    "batchdownload.py",
    "laz_to_las.py",
    "delete_empty_files.py",
    "las_to_dem.py",
    "mosaic_dem.py",
    "wbt_hydrology.py",
    "stream_extraction_wbt.py",
    "delineate_watersheds.py",
    "calculate_ksn.py",
    "plot_stream_profiles.py"
]

def run_script(script_name):
    print(f"\n{'='*40}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*40}")

    script_path = os.path.join(SCRIPTS_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"ERROR: Script not found at expected path: {script_path}")
        return False

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        duration = time.time() - start_time
        print(f"SUCCESS: {script_name} finished in {duration:.2f} seconds.")
        return True

    except subprocess.CalledProcessError as e:
        print(f"ERROR: {script_name} failed with exit code {e.returncode}")
        return False

def main():
    print(f"Scripts directory: {SCRIPTS_DIR}")
    for script in SCRIPTS_TO_RUN:
        success = run_script(script)
        if not success:
            print("\nPIPELINE HALTED: A critical error occurred.")
            break
    else:
        print("\nALL SCRIPTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()