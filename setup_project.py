import os
import sys
import shutil
import subprocess
from pathlib import Path

try:
    import config
    print("[OK] Config file detected.")
except ImportError:
    print("[ERROR] config.py not found! Ensure it is in the same folder as this script.")
    sys.exit(1)

def run_setup():
    print("==========================================")
    print("   SAN BERNARDINO Ksn PROJECT SETUP")
    print("==========================================\n")

    # 2. Build the folders defined in config.py
    print("[1/3] Verifying Directory Structure...")
    # List of directory variables from your config
    required_dirs = [
        config.DATA_RAW, 
        config.DATA_PROCESSED,
        config.DATA_SCRATCH,
        config.DATA_SCRATCH_DEMS,
        config.DATA_DEM_MOSAIC,
        config.DATA_SCRATCH_WBT,
        config.DATA_SCRATCH_WATERSHEDS,
        config.DATA_STREAMS,
        config.DATA_WATERSHEDS,
        config.DATA_KSN,
        config.FIGURES_DIR
    ]
    
    for folder in required_dirs:
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            print(f"      Created: {folder.relative_to(config.ROOT_DIR)}")
        else:
            print(f"      Exists:  {folder.relative_to(config.ROOT_DIR)}")

    # 3. Validate External Executables
    print("\n[2/3] Validating External Tools...")
    
    # Check WhiteboxTools
    if config.WBT_EXE.exists():
        try:
            # Attempt a version check via subprocess
            result = subprocess.run([str(config.WBT_EXE), "--version"], 
                                    capture_output=True, text=True, check=True)
            print(f"      [PASS] WhiteboxTools: {result.stdout.strip()}")
        except Exception as e:
            print(f"      [FAIL] WhiteboxTools found at {config.WBT_EXE} but failed to run: {e}")
    else:
        print(f"      [MISSING] WhiteboxTools not found at: {config.WBT_EXE}")

    # Check LAStools
    lastools_check = config.LASTOOLS_BIN / "laszip.exe"
    if lastools_check.exists():
        print(f"      [PASS] LAStools located in: {config.LASTOOLS_BIN}")
    else:
        print(f"      [MISSING] LAStools (laszip.exe) not found at: {lastools_check}")

    # 4. Check ArcGIS Environment
    print("\n[3/3] Checking GIS Environment...")
    try:
        import arcpy
        print(f"      [PASS] ArcPy found ({sys.executable})")
        if arcpy.CheckExtension("Spatial") == "Available":
            print("      [PASS] Spatial Analyst Extension is available.")
        else:
            print("      [WARN] Spatial Analyst NOT available. Check your license.")
    except ImportError:
        print("      [FAIL] ArcPy not found. Are you in the ArcGIS Python Command Prompt?")

    print("\n==========================================")
    print(" Setup Complete. Check the [MISSING] items above.")
    print("==========================================")

if __name__ == "__main__":
    run_setup()