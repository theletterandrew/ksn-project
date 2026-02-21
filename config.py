import os
from pathlib import Path

# Base Paths
ROOT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
FIGURES_DIR = ROOT_DIR / "figures"

# Data Paths
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
DATA_DEM_MOSAIC = ROOT_DIR / "data" / "mosaic"
DATA_STREAMS = ROOT_DIR / "data" / "streams"
DATA_WATERSHEDS = ROOT_DIR / "data" / "watersheds"
DATA_KSN = ROOT_DIR / "data" / "ksn"

# Scratch Data Paths
DATA_SCRATCH = ROOT_DIR / "data" / "scratch"
DATA_SCRATCH_DEMS = DATA_SCRATCH / "DEMs"
DATA_SCRATCH_WBT = DATA_SCRATCH / "WBT"
DATA_SCRATCH_WATERSHEDS = DATA_SCRATCH / "watersheds"

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
    # A small 500m x 500m patch for testing logic
    # Coordinates in EPSG:3857 (Web Mercator)
    BOUNDS_STR = "([-13100000, -13095000],[3980000, 3985000])"
    print("--- RUNNING IN TEST MODE (Small Area) ---")
else:
    # The full study area bounds
    BOUNDS_STR = "([-13035749.581531966,-12973917.047710635],[4018953.87470956,4080431.0491411127])"

# --- DOWNLOAD PARAMETERS ---
# Enter your download tile size in meters
TILE_SIZE = 5000

# Enter the amount of overlap between tiles in meters
OVERLAP = 2000

# Enter the LiDAR resolution you'd like to download in meters
RES = 2.0

# --- EXTRACTION PARAMETERS ---
# Number of parallel extractions for extracting LAZ to LAS
MAX_WORKERS = 4

# Stream threshold (in pixels)
STREAM_THRESHOLD = 1000000    # cells (~4 km² at 2m resolution)

# --- DELETE EMPTY FILES PARAMETERS ---
MIN_TILE_SIZE_KB = 1

# --- WATERSHED PARAMETERS ---
# Watershed minimum drainage area threshold (~40 km² at 2m resolution)
MIN_DRAINAGE_AREA_CELLS = 10000000

# Pour points are snapped to the highest flow accumulation cell within
# this distance to ensure they land exactly on the stream
SNAP_DISTANCE = 50

# --- KSN ANALYSIS PARAMETERS ---
MIN_DRAINAGE_AREA_M2 = 1000000           # Min drainage area for stream extraction (1 km²)
REFERENCE_CONCAVITY  = 0.45          # Reference concavity index (theta_ref)
SMOOTHING_WINDOW     = 5             # Window size (cells) for gradient smoothing
SAMPLE_DISTANCE      = 50            # Sample points every N meters along streams