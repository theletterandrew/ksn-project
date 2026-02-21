import subprocess
import sys
import time
import os
from pathlib import Path

def sanitize_path():
    """Remove LAStools from PATH to prevent its GDAL DLL from conflicting with conda-forge's."""
    paths = os.environ["PATH"].split(os.pathsep)
    cleaned = [p for p in paths if "LAStools" not in p]
    os.environ["PATH"] = os.pathsep.join(cleaned)

    # Explicitly prepend the ksn_env DLL directory so rasterio finds the right GDAL
    ksn_bin = Path(KSNENV_PYTHON).parent.parent / "Library" / "bin"
    os.environ["PATH"] = str(ksn_bin) + os.pathsep + os.environ["PATH"]

def find_ksn_python():
    """Find the ksn_env Python executable via conda."""
    result = subprocess.run(
        ["conda", "run", "-n", "ksn_env", "python", "-c", "import sys; print(sys.executable)"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return result.stdout.strip()
    raise EnvironmentError("Could not locate ksn_env. Has it been created yet?")

def find_arcgis_python():
    """Find arcgispro-py3 Python executable by querying conda directly."""
    result = subprocess.run(
        ["conda", "run", "-n", "arcgispro-py3", "python", "-c", "import sys; print(sys.executable)"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return result.stdout.strip()
    raise EnvironmentError("Could not locate arcgispro-py3. Is ArcGIS Pro installed?")

KSNENV_PYTHON = find_ksn_python()
ARCGIS_PYTHON = find_arcgis_python()

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")


SCRIPTS_TO_RUN = [
    ("batchdownload.py",           KSNENV_PYTHON),
    ("laz_to_las.py",              KSNENV_PYTHON),
    ("delete_empty_files.py",      KSNENV_PYTHON),
    ("las_to_dem.py",              KSNENV_PYTHON),
    ("mosaic_dem.py",              ARCGIS_PYTHON),
    ("wbt_hydrology.py",           KSNENV_PYTHON),
    ("stream_extraction_wbt.py",   ARCGIS_PYTHON),
    ("delineate_watersheds.py",    ARCGIS_PYTHON),
    ("clip_watersheds.py",         ARCGIS_PYTHON),
    ("calculate_ksn.py",           KSNENV_PYTHON),
    ("plot_stream_profiles.py",    KSNENV_PYTHON),
]

def run_script(script_name, python_exec):
    print(f"\n{'='*40}")
    print(f"RUNNING: {script_name}")
    print(f"PYTHON:  {python_exec}")
    print(f"{'='*40}")

    script_path = os.path.join(SCRIPTS_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"ERROR: Script not found at expected path: {script_path}")
        return False
    
    if not os.path.exists(python_exec):
        print(f"ERROR: Python executable not found: {python_exec}")
        return False

    start_time = time.time()

    try:
        subprocess.run(
            [python_exec, script_path],
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
    sanitize_path()
    print(f"Scripts directory: {SCRIPTS_DIR}")
    for script, python_exec in SCRIPTS_TO_RUN:
        success = run_script(script, python_exec)
        if not success:
            print("\nPIPELINE HALTED: A critical error occurred.")
            break
    else:
        print("\nALL SCRIPTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()