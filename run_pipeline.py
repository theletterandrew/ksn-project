import subprocess
import sys
import time

# List your scripts in the exact order they should run
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
    
    start_time = time.time()
    
    try:
        # sys.executable ensures it uses the same Python environment/conda path
        result = subprocess.run([sys.executable, script_name], check=True)
        
        duration = time.time() - start_time
        print(f"SUCCESS: {script_name} finished in {duration:.2f} seconds.")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {script_name} failed with exit code {e.returncode}")
        return False

def main():
    for script in SCRIPTS_TO_RUN:
        success = run_script(script)
        if not success:
            print("\nPIPELINE HALTED: A critical error occurred.")
            break
    else:
        print("\nALL SCRIPTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()