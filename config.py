from pathlib import Path
import os

_conda_prefix = os.environ.get("CONDA_PREFIX", "")
if _conda_prefix:
    _dll_dir = os.path.join(_conda_prefix, "Library", "bin")
    if os.path.exists(_dll_dir):
        os.add_dll_directory(_dll_dir)
    
    # Force pyproj to use the correct PROJ data directory
    _proj_data = os.path.join(_conda_prefix, "Library", "share", "proj")
    if os.path.exists(_proj_data):
        os.environ["PROJ_DATA"] = _proj_data
        os.environ["PROJ_LIB"] = _proj_data

os.environ["GDAL_DRIVER_PATH"] = ""

# Base Paths
ROOT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
FIGURES_DIR = ROOT_DIR / "figures"

# Data Paths
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
DATA_DEM_MOSAIC = ROOT_DIR / "data" / "mosaic"
DATA_DEM_RAW_MOSAIC = DATA_DEM_MOSAIC / "dem_mosaic.tif"
DATA_DEM_MOSAIC_WBT = DATA_DEM_MOSAIC / "dem_mosaic_wbt.tif"
DATA_STREAMS = ROOT_DIR / "data" / "streams"
DATA_WATERSHEDS = ROOT_DIR / "data" / "watersheds"
DATA_KSN = ROOT_DIR / "data" / "ksn"

# Scratch Data Paths
DATA_SCRATCH = ROOT_DIR / "data" / "scratch"
DATA_SCRATCH_DEMS = DATA_SCRATCH / "DEMs"
DATA_SCRATCH_WBT = DATA_SCRATCH / "WBT"
DATA_SCRATCH_WATERSHEDS = DATA_SCRATCH / "watersheds"
DATA_DEM_CONDITIONED = DATA_SCRATCH_WBT / "dem_filled.tif"

# Tool Paths
WBT_EXE = ROOT_DIR / "bin" / "WBT" / "whitebox_tools.exe"
LASZIP_EXE = ROOT_DIR / "bin" / "LAStools" / "bin" / "laszip.exe"
LASTOOLS_BIN = ROOT_DIR / "bin" / "LAStools" / "bin"

# Project Settings
SEARCH_RADIUS = 15

# --- CONFIGURATION ---
# GLOBAL SWITCH
TEST_RUN = True # Set to False for the full San Bernardino study area

# Ensure this is your verified URL
EPT_URL = "http://usgs-lidar-public.s3.amazonaws.com/USGS_LPC_CA_SoCal_Wildfires_B1_2018_LAS_2019/ept.json"

# --- DATA SOURCE LOGIC ---
if TEST_RUN:
    # Santa Ana Mountains — Harding Canyon / Modjeska Peak / Santiago Canyon
    # Coordinates in EPSG:3857 (Web Mercator), ~16x16 km mountain block
    BOUNDS_STR = "([-13018226.004, -13013226.004],[4042885.562, 4047885.562])"
    print("--- RUNNING IN TEST MODE (Small Area) ---")
else:
    # The full study area bounds
    BOUNDS_STR = "([-13035749.581531966,-12973917.047710635],[4018953.87470956,4080431.0491411127])"

# --- DOWNLOAD PARAMETERS ---
# Enter your download tile size in meters
TILE_SIZE = 6000 if TEST_RUN else 5000

# Enter the amount of overlap between tiles in meters
OVERLAP = 200

# Enter the LiDAR resolution you'd like to download in meters
RES = 2.0

# --- EXTRACTION PARAMETERS ---
# Number of parallel extractions for extracting LAZ to LAS
MAX_WORKERS = 4

# Stream threshold (in pixels)
MIN_DRAINAGE_AREA_CELLS = 900000 if TEST_RUN else 10000000    # cells (~4 km² at 2m resolution)

# Length (m) of minimum tributary length. Filters out short stream segments.
MIN_STREAM_LENGTH_M = 500         

# Number of border cells to blank on all four edges before thresholding.
# Edge cells drain off-raster in D8, accumulating spurious flow.
BORDER_CELLS = 3

# Minimum distance separating the outlet points (in meters)
MIN_OUTLET_SEPARATION = 2000

# --- DELETE EMPTY FILES PARAMETERS ---
MIN_TILE_SIZE_KB = 1

# --- HYDROLOGY PARAMETERS ---
# Maximum breach distance in cells for BreachDepressionsLeastCost
# At 2m resolution: 100 cells = 200m (suitable for small test areas)
#                   1000 cells = 2km  (suitable for full study area)
WBT_BREACH_DIST = 100 if TEST_RUN else 1000

# --- WATERSHED PARAMETERS ---
# Watershed minimum drainage area threshold
# Test dataset: (~0.4 km^2 at 2m resolution)
# Full dataset: (~40 km² at 2m resolution)
MIN_WATERSHED_AREA = 200000 if TEST_RUN else 10000000

# Pour points are snapped to the highest flow accumulation cell within
# this distance to ensure they land exactly on the stream
SNAP_DISTANCE = 50

# --- STREAM EXTRACTION PARAMETERS ---
STREAM_THRESHOLD        = 50000 if TEST_RUN else 1000000

# --- KSN ANALYSIS PARAMETERS ---
MIN_DRAINAGE_AREA_M2 = 200000 if TEST_RUN else 1000000
REFERENCE_CONCAVITY  = 0.45          # Reference concavity index (theta_ref)
SMOOTHING_WINDOW     = 5             # Window size (cells) for gradient smoothing
SAMPLE_DISTANCE      = 50            # Sample points every N meters along streams
MIN_TRIBUTARY_LENGTH_M = 500         # Length (m) of minimum tributary length. Filters out short stream segments.