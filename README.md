# Lidar-to-Ksn Pipeline for San Bernardino Mountains

Automated workflow for calculating normalized channel steepness index (ksn) from lidar data for the San Bernardino Mountains, Southern California.

## Study Area

**Bounding Box:** 33.93°N to 34.38°N and 117.10°W to 116.55°W (WGS84)  
**Area:** ~3,800 km²  
**Location:** San Bernardino Mountains, Southern California

## Data Source

**Dataset:** USGS_LPC_CA_SoCal_Wildfires_B1_2018_LAS_2019  
**Format:** LAZ (compressed LAS)  
**Acquisition Date:** 2018  
**Publication Date:** 2019  
**Source:** USGS AWS EPT streaming service  
**Metadata:** https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/laz/geoid18/9003/index.html

## Pipeline Overview

```
LAZ Files (441 tiles, 152 GB)
    ↓
LAS Extraction & Filtering
    ↓
DEM Generation (274 tiles, 2m resolution)
    ↓
DEM Mosaicking
    ↓
Hydrological Conditioning (WhiteboxTools)
    ↓
Stream Network Extraction
    ↓
Watershed Delineation
    ↓
Ksn Calculation
    ↓
Visualization & Analysis
```

---

## Quick Start Setup

### 1. Download Project

Clone the repository or download as ZIP:
```bash
git clone [repository_url]
```

Or download ZIP from GitHub and extract to your desired location.

### 2. Install WhiteboxTools

1. Download the binary for your OS from: https://www.whiteboxgeo.com/
2. Extract the ZIP file
3. Copy the contents of the extracted "WBT" folder into:
   ```
   SanBernardino_Ksn_Project/bin/WBT/
   ```
4. **Verification:** You should see `whitebox_tools.exe` inside your `bin/WBT` folder

### 3. Install LAStools

1. Download the ZIP from: https://rapidlasso.de/
2. Extract the ZIP file
3. Copy the contents of the extracted "LAStools" folder to:
   ```
   SanBernardino_Ksn_Project/bin/LAStools/
   ```
4. **Verification:** You should see a `bin` folder inside `bin/LAStools` (e.g., `bin/LAStools/bin/laszip.exe`)

### 4. Run Project Initialization

**Important:** Open the **ArcGIS Python Command Prompt** and navigate to the project folder:
```bash
cd C:\Path\To\Your\SanBernardino_Ksn_Project
```

Run the setup batch file:
```bash
setup.bat
```

**Verification:** You should see a message saying that the setup is complete.

### 5. Run the Pipeline

Execute the main pipeline script:
```bash
python run_pipeline.py
```

**Verification:** You should see a message saying "ALL SCRIPTS COMPLETED SUCCESSFULLY!" at the end of the process.

---

## Detailed Methods

### 1. Downloading and Preprocessing

**Script:** `batchdownload.py`

- Downloaded lidar data in LAZ format from USGS AWS server
- Study area divided into 5 km × 5 km tiles with 2 km overlap
- Data requested at 2 m resolution (server-side resampling by EPT service)
- Point cloud density: ~1.5 pts/m² (~38 million points per tile)
- Total: 441 tiles, 152 GB

**Coordinate Systems:**
- **Horizontal (input):** EPSG:3857 (WGS 84 / Pseudo-Mercator)
  - Note: Original acquisition was NAD83(NSRS2007) / UTM Zone 10N
  - Data was reprojected to EPSG:3857 by USGS during EPT ingestion
- **Vertical:** NAVD88 (GEOID18), meters
  - Not stored in LAZ headers; documented on NOAA landing page

**Decompression:**
- **Script:** `laz_to_las.py`
- Tool: LASzip
- Decompressed LAZ → LAS for compatibility with ArcGIS Pro

**Filtering:**
- **Script:** `delete_empty_files.py`
- Removed empty tiles (< 10 KB)
- Result: 274 valid tiles for processing

### 2. DEM Creation

**Script:** `las_to_dem.py`

**Libraries:** `laspy`, `scipy`, `rasterio`, `pyproj`

**Process:**
1. Reprojected point coordinates from EPSG:3857 → EPSG:26911 (NAD83 / UTM Zone 11N)
   - UTM Zone 11N is appropriate for Southern California geomorphic analysis
2. Generated 2 m resolution DEMs using mean gridding
3. Applied inverse distance weighted (IDW) interpolation to fill empty cells
4. Saved as float32 GeoTIFFs with deflate compression

**Output Specifications:**
- Resolution: 2 m
- CRS: EPSG:26911 (NAD83 / UTM Zone 11N)
- Vertical datum: NAVD88 (GEOID18), meters
- Format: Float32 GeoTIFF

### 3. DEM Mosaicking

**Script:** `mosaic_dem.py`

**Tool:** ArcGIS Pro 3.6 `MosaicToNewRaster`

- Combined 274 DEMs into single seamless mosaic
- Mosaic method: "LAST" (last tile's value used in overlap zones)
- Output: `dem_mosaic.tif`

### 4. Hydrological Conditioning

**Script:** `wbt_hydrology.py`

**Tool:** WhiteboxTools (called as subprocess from Python)

**Process:**
1. **Breach depressions** — `BreachDepressionsLeastCost`
   - Max breach distance: 1000 cells (2 km)
   - Preserves natural channel geometry better than filling alone
2. **Fill remaining depressions** — `FillDepressions`
   - Handles any depressions that couldn't be breached
3. **Flow direction** — `D8Pointer`
   - D8 algorithm for flow routing
4. **Flow accumulation** — `D8FlowAccumulation`
   - Counts upstream contributing cells

**Outputs:**
- `dem_breached.tif`
- `dem_filled.tif`
- `flow_direction.tif`
- `flow_accumulation.tif`

### 5. Stream Network Extraction

**Script:** `stream_extraction_wbt.py`

**Tool:** ArcGIS Pro Spatial Analyst `StreamToFeature`

**Process:**
1. Applied threshold to flow accumulation raster
   - Threshold: 1,000,000 cells (~4 km² at 2m resolution)
   - Prevents overly dense network while capturing major channels
2. Created binary stream raster (stream vs non-stream cells)
3. Converted stream raster to vector polylines using `StreamToFeature`
   - Used D8 flow direction for connectivity tracing
   - Applied geometry simplification

**Output:** `streams_connected.shp` — fully connected stream network

### 6. Watershed Delineation

**Script:** `delineate_watersheds.py`

**Tool:** ArcGIS Pro Spatial Analyst `Watershed`, `SnapPourPoint`

**Process:**
1. Identified stream endpoints from `streams_connected.shp`
2. Extracted flow accumulation values at each endpoint
3. Selected outlets above minimum drainage area threshold
   - Threshold: ~40 km² (default)
   - Adjustable based on study objectives
4. Snapped pour points to highest flow accumulation cells within search radius
   - Ensures points land exactly on streams
5. Delineated contributing watershed for each pour point using `Watershed` tool
6. Converted watershed raster to polygons

**Output:** `watersheds.shp` — polygon shapefile with watershed area (km²) attribute

### 7. Watershed DEM Clipping

**Script:** `clip_watersheds.py`

**Tool:** ArcGIS Pro Spatial Analyst `ExtractByMask`

- Clipped full DEM mosaic to each individual watershed polygon
- Produces manageable-sized DEMs suitable for ksn analysis
- Output naming: `watershed_X.tif` (where X = watershed ID)
- Maintains 2m resolution, EPSG:26911, NAVD88 (GEOID18)

### 8. Ksn Calculation

**Script:** `calculate_ksn.py`

**Libraries:** `numpy`, `scipy`, `rasterio`, `geopandas`

**Parameters:**
- **Stream threshold:** 1,000,000 cells (~4 km²)
- **Reference concavity (θ):** 0.45 (standard for bedrock channels)
- **Gradient smoothing window:** 5 cells (10 m)
- **Sample interval:** 50 m along streams

**Process:**
1. Loaded watershed DEM and flow accumulation raster
2. Extracted stream mask (cells above threshold)
3. Calculated slope using smoothed gradient
   - Applied moving window mean filter to reduce lidar microtopography noise
   - Used central differences for gradient calculation
4. Calculated drainage area from flow accumulation (cells × cell size²)
5. Computed ksn using slope-area relationship:
   ```
   ksn = slope / (drainage_area^-θ)
   ```
6. Sampled points every 50 m along streams
7. Exported point shapefile with attributes:
   - `ksn` — normalized channel steepness index
   - `slope` — gradient (m/m)
   - `area_km2` — drainage area (km²)

**Output:** `watershed_X_ksn.shp` for each watershed

### 9. Visualization

**Script:** `plot_stream_profiles.py`

**Libraries:** `matplotlib`, `geopandas`, `rasterio`

**Generated Figures:**
- Stream profile: elevation vs distance downstream
- Points color-coded by ksn value
- Statistics box: mean ksn, elevation range, profile length
- Colorbar: ksn scale
- Resolution: 300 DPI PNG

**Output:** `watershed_X_profile.png` for each watershed

---

## Software Requirements

### Python Environments

**demenv** (for DEM creation, ksn calculation, visualization):
```bash
conda create -n demenv python=3.11
conda activate demenv
conda install -c conda-forge laspy lazrs-python scipy rasterio pyproj geopandas matplotlib
```

**arcgispro-py3** (for hydrological processing, mosaicking):
- ArcGIS Pro 3.6 with Spatial Analyst extension
- Pre-installed with ArcGIS Pro

### External Tools

**WhiteboxTools:**
- Download from: https://www.whiteboxgeo.com/download-whiteboxtools/
- Add to system PATH
- Free, open-source hydrological analysis toolbox

**LASzip:**
- Included with LAStools
- Used for LAZ decompression

---

## File Structure

```
E:\LiDAR\Scoped\
├── ground_tiles\          # Original LAZ files (441 tiles, 152 GB)
├── Extracted\             # Decompressed LAS files (deleted after DEM creation)
├── DEMs\                  # Individual DEM tiles (274 tiles)
│   └── Mosaic\           # Full DEM mosaic
├── WBT\                   # WhiteboxTools outputs
│   ├── dem_breached.tif
│   ├── dem_filled.tif
│   ├── flow_direction.tif
│   └── flow_accumulation.tif
├── Streams\
│   └── WBT\
│       └── streams_connected.shp  # Connected stream network
├── Watersheds\
│   ├── watersheds.shp             # Watershed polygons
│   └── DEMs\                      # Individual watershed DEMs
│       ├── watershed_1.tif
│       ├── watershed_2.tif
│       └── ...
├── Ksn\                           # Ksn point shapefiles
│   ├── watershed_1_ksn.shp
│   ├── watershed_2_ksn.shp
│   └── ...
└── Figures\                       # Stream profile plots
    ├── watershed_1_profile.png
    ├── watershed_2_profile.png
    └── ...
```

---

## Scripts

All scripts are documented with usage instructions in their headers.

### Data Acquisition & Preprocessing
- `batchdownload.py` — Download LAZ tiles from USGS AWS
- `laz_to_las.py` — Decompress LAZ to LAS
- `delete_empty_files.py` — Remove empty tiles

### DEM Generation
- `las_to_dem.py` — Batch LAS to GeoTIFF conversion
- `mosaic_dem.py` — Mosaic DEMs into single raster

### Hydrological Analysis
- `wbt_hydrology.py` — WhiteboxTools hydrology pipeline
- `stream_extraction_wbt.py` — Extract stream network
- `delineate_watersheds.py` — Automated watershed delineation
- `clip_watersheds.py` — Clip DEMs to watersheds

### Ksn Analysis
- `calculate_ksn.py` — Batch ksn calculation for watersheds
- `plot_stream_profiles.py` — Generate stream profile figures

### Utilities
- `check_classification.py` — Diagnostic: point classification
- `check_laz_crs.py` — Diagnostic: CRS verification
- `cleanup_bad_tifs.py` — Identify corrupted GeoTIFFs
- `rename_tiles.py` — Rename files to avoid arcpy length limits

---

## Key Design Decisions

### Why 2 km overlap?
The 2 km overlap between tiles ensures proper flow routing near tile boundaries. Edge effects in hydrological processing can create artifacts; the overlap provides a buffer zone that gets trimmed during final mosaicking.

### Why WhiteboxTools instead of ArcGIS Pro for hydrology?
WhiteboxTools streams data off disk rather than loading entire rasters into RAM, making it suitable for the full 3,800 km² DEM mosaic (~950 million pixels at 2m resolution). ArcGIS Pro would struggle with memory on datasets this large.

### Why breach before fill?
Breaching carves a narrow path through depressions rather than raising surrounding terrain, better preserving the natural DEM surface. This is especially important for high-resolution lidar DEMs where topography is accurately captured.

### Why NAD83 / UTM Zone 11N?
Although the original data is in EPSG:3857 (Web Mercator), this projection introduces significant distortion (~20% at 34°N) and is inappropriate for geomorphic analysis requiring accurate distances and areas. UTM Zone 11N provides minimal distortion for the study area longitude (~117°W).

### Why calculate ksn directly instead of using TopoToolbox?
The Python version of TopoToolbox is less mature than the MATLAB version and lacks complete documentation for ksn workflows. Direct calculation using numpy/scipy provides full control over the methodology and avoids dependency on an unstable library.

---

## Performance Notes

Approximate processing times on a system with:
- CPU: Modern multi-core processor
- RAM: 32 GB
- Storage: M.2 NVMe SSD

| Step | Time (274 tiles) |
|------|------------------|
| LAZ decompression | ~30 min |
| DEM creation | ~12 hours |
| DEM mosaicking | ~30 min |
| WhiteboxTools hydrology | ~1-2 hours |
| Stream extraction | ~5 min |
| Watershed delineation | ~10 min |
| Watershed DEM clipping | ~20 min |
| Ksn calculation | ~30 min |
| Figure generation | ~10 min |

**Total:** ~16-17 hours

---

## Citation

If using this pipeline, please cite the original data source:

**Lidar Data:**
> NOAA Office for Coastal Management. 2018 USGS Lidar: Southern California Wildfires. Accessed [date]. Available: https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/laz/geoid18/9003/index.html

**WhiteboxTools:**
> Lindsay, J.B. (2016). Whitebox GAT: A case study in geomorphometric analysis. *Computers & Geosciences*, 95: 75-84. DOI: 10.1016/j.cageo.2016.07.003

---

## Troubleshooting

### "ERROR 010240: Could not save raster dataset"
- Cause: Arcpy filename length limit (13 characters for certain operations)
- Solution: Use shorter filenames (e.g., `fac_gt_X.tif` instead of `fac_ground_tile_X.tif`)

### "ImportError: DLL load failed" (shapely/rasterio)
- Cause: DLL conflicts between conda packages
- Solution: Create fresh environment with strict channel priority:
  ```bash
  conda create -n geoenv python=3.11
  conda config --env --set channel_priority strict
  conda install -c conda-forge rasterio shapely geopandas
  ```

### Stream network has gaps at tile boundaries
- Cause: Flow accumulation was computed per-tile rather than on full mosaic
- Solution: Must run hydrology on full mosaicked DEM (WhiteboxTools approach)

### Out of memory errors
- Cause: Attempting to process full mosaic with memory-intensive tools
- Solution: Use WhiteboxTools which streams data off disk

---

## Future Improvements

- Implement D-infinity flow routing for more accurate flow direction
- Add support for custom drainage area thresholds per watershed
- Integrate with chi analysis for comparison with ksn
- Automate knickpoint detection along profiles
- Add batch export to publication-ready figures

---

## Contact

For questions about this pipeline, please open an issue on the repository.

## License

Scripts provided as-is for research and educational purposes.
