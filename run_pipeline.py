import subprocess
import sys
import time
import os


def sanitize_env():
    """Return a copy of the environment safe for running ksn_env scripts."""
    env = os.environ.copy()

    # Strip LAStools from PATH to prevent its gdal.dll from conflicting
    env["PATH"] = os.pathsep.join(
        p for p in env["PATH"].split(os.pathsep) if "LAStools" not in p
    )

    # Suppress ArcGIS GDAL plugins
    env["GDAL_DRIVER_PATH"] = ""

    # Force pyproj/GDAL to use conda-forge's PROJ data, not a system install
    conda_prefix = env.get("CONDA_PREFIX", "")
    if conda_prefix:
        proj_data = os.path.join(conda_prefix, "Library", "share", "proj")
        if os.path.exists(proj_data):
            env["PROJ_DATA"] = proj_data
            env["PROJ_LIB"]  = proj_data

    return env


SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")

SCRIPTS_TO_RUN = [
    # "batchdownload.py",
    # "laz_to_las.py",
    # "las_diagnostics.py",
    # "delete_empty_files.py",
    # "las_to_dem.py",
    # "mosaic_dem.py",
    # "wbt_hydrology.py",
    # "stream_extraction_wbt.py",
    # "delineate_watersheds.py",
    # "clip_watersheds.py",
    # "extract_longest_branches.py",
    "stream_and_watersheds.py",
    # "test_watershed_dtype.py",
    # "calculate_ksn.py",
    # "plot_stream_profiles.py"
]


def run_script(script_name: str, env: dict) -> bool:
    print(f"\n{'='*40}")
    print(f"RUNNING: {script_name}")
    print(f"PYTHON:  {sys.executable}")
    print(f"{'='*40}")

    script_path = os.path.join(SCRIPTS_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"ERROR: Script not found at expected path: {script_path}")
        return False

    start_time = time.time()

    try:
        subprocess.run(
            [sys.executable, script_path],
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env,
        )
        duration = time.time() - start_time
        print(f"SUCCESS: {script_name} finished in {duration:.2f} seconds.")
        return True

    except subprocess.CalledProcessError as e:
        print(f"ERROR: {script_name} failed with exit code {e.returncode}")
        return False


def main():
    # Verify we're running inside ksn_env
    active_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if active_env != "ksn_env":
        print(f"WARNING: Expected to be running in 'ksn_env', but active env is '{active_env or '(none)'}'.")
        print("Activate it first with:  conda activate ksn_env")
        print("Continuing anyway...\n")

    clean_env = sanitize_env()

    print(f"Python:           {sys.executable}")
    print(f"Scripts directory: {SCRIPTS_DIR}")

    for script in SCRIPTS_TO_RUN:
        success = run_script(script, clean_env)
        if not success:
            print("\nPIPELINE HALTED: A critical error occurred.")
            sys.exit(1)

    print("\nALL SCRIPTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
