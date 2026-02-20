"""
plot_stream_profiles.py
-----------------------
Generates stream profile plots for each watershed showing elevation vs
distance downstream with ksn values color-coded along the profile.

For each watershed, creates a figure with:
    - Main plot: elevation profile with points colored by ksn
    - Colorbar: ksn scale
    - Statistics: mean ksn, elevation range, profile length

USAGE:
    1. Edit the paths in the CONFIG section below.
    2. Run from an environment with matplotlib:
       conda activate demenv
       python plot_stream_profiles.py

Requirements:
    conda install -c conda-forge matplotlib geopandas numpy scipy
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches

# Calculate the path to the project root (one level up from scripts/)
root_dir = Path(__file__).resolve().parent.parent

# Add the root directory to sys.path so Python can find config.py
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import config

# =============================================================================
# CONFIG — Edit these before running
# =============================================================================

KSN_DIR     = config.DATA_KSN             # Folder with *_ksn.shp files
OUTPUT_DIR  = config.FIGURES_DIR          # Output folder for figures

# Plot styling
FIGURE_SIZE    = (12, 6)      # Figure size in inches (width, height)
POINT_SIZE     = 20           # Size of points on profile
COLORMAP       = 'viridis'    # Colormap for ksn values (viridis, plasma, coolwarm)
DPI            = 300          # Resolution for saved figures

# Ksn colorbar range (set to None for auto-scaling per watershed)
KSN_VMIN = None    # Minimum ksn for colorbar (None = auto)
KSN_VMAX = None    # Maximum ksn for colorbar (None = auto)

# =============================================================================
# END CONFIG — No edits needed below this line
# =============================================================================


def setup_logging(output_dir: Path) -> logging.Logger:
    log_path = output_dir / "plot_stream_profiles.log"
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


def calculate_downstream_distance(points_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Calculate cumulative distance downstream from the uppermost point.
    Assumes points are already sorted by elevation (highest to lowest).
    Returns array of distances in meters.
    """
    coords = np.array([(p.x, p.y) for p in points_gdf.geometry])
    
    # Calculate distances between consecutive points
    diffs = np.diff(coords, axis=0)
    distances = np.sqrt((diffs ** 2).sum(axis=1))
    
    # Cumulative distance from upstream
    cum_dist = np.concatenate([[0], np.cumsum(distances)])
    
    return cum_dist


def extract_elevation_from_dem(points_gdf: gpd.GeoDataFrame, 
                                dem_path: Path) -> np.ndarray:
    """
    Extract elevation values at each point location from the DEM.
    Returns array of elevations in meters.
    """
    import rasterio
    
    with rasterio.open(str(dem_path)) as src:
        # Extract coordinates
        coords = [(p.x, p.y) for p in points_gdf.geometry]
        
        # Sample DEM at point locations
        elevations = [val[0] for val in src.sample(coords)]
    
    return np.array(elevations)


def plot_stream_profile(ksn_shp: Path, dem_path: Path, output_dir: Path,
                       logger: logging.Logger) -> tuple[bool, str]:
    """
    Creates a stream profile plot for a single watershed.
    Returns (success, output_path).
    """
    watershed_id = ksn_shp.stem.replace('_ksn', '')  # e.g. "watershed_1"
    out_fig      = output_dir / f"{watershed_id}_profile.png"
    
    # Skip if already exists
    if out_fig.exists():
        return (True, str(out_fig))
    
    try:
        # Load ksn shapefile
        gdf = gpd.read_file(str(ksn_shp))
        
        if len(gdf) == 0:
            logger.warning(f"  No points in shapefile — skipping")
            return (False, "")
        
        # Extract elevation from DEM
        logger.info(f"  Extracting elevations from DEM...")
        elevations = extract_elevation_from_dem(gdf, dem_path)
        
        # Sort by elevation (highest to lowest = upstream to downstream)
        sort_idx   = np.argsort(elevations)[::-1]
        gdf_sorted = gdf.iloc[sort_idx].reset_index(drop=True)
        elev_sorted = elevations[sort_idx]
        
        # Calculate downstream distance
        logger.info(f"  Computing downstream distances...")
        distances = calculate_downstream_distance(gdf_sorted)
        
        # Extract ksn values
        ksn_values = gdf_sorted['ksn'].values
        
        # Create figure
        logger.info(f"  Generating plot...")
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        # Determine ksn color range
        vmin = KSN_VMIN if KSN_VMIN is not None else np.percentile(ksn_values, 5)
        vmax = KSN_VMAX if KSN_VMAX is not None else np.percentile(ksn_values, 95)
        
        # Create scatter plot with ksn colors
        scatter = ax.scatter(
            distances / 1000,  # Convert to km
            elev_sorted,
            c=ksn_values,
            s=POINT_SIZE,
            cmap=COLORMAP,
            vmin=vmin,
            vmax=vmax,
            alpha=0.8,
            edgecolors='none'
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='ksn')
        
        # Labels and title
        ax.set_xlabel('Distance downstream (km)', fontsize=12)
        ax.set_ylabel('Elevation (m)', fontsize=12)
        ax.set_title(f'Stream Profile - {watershed_id}', fontsize=14, fontweight='bold')
        
        # Add statistics text box
        stats_text = (
            f"Mean ksn: {np.mean(ksn_values):.1f}\n"
            f"Std ksn: {np.std(ksn_values):.1f}\n"
            f"Elevation range: {elev_sorted.max():.0f} - {elev_sorted.min():.0f} m\n"
            f"Profile length: {distances.max()/1000:.1f} km\n"
            f"Points: {len(gdf_sorted)}"
        )
        
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(str(out_fig), dpi=DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved: {out_fig.name}")
        return (True, str(out_fig))
        
    except Exception as e:
        logger.error(f"  Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        if out_fig.exists():
            try:
                out_fig.unlink()
            except Exception:
                pass
        return (False, "")


def main():
    ksn_dir    = Path(KSN_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    
    # Find all ksn shapefiles and their corresponding DEMs
    ksn_files = sorted(ksn_dir.glob("watershed_*_ksn.shp"))
    if not ksn_files:
        logger.error(f"No watershed_*_ksn.shp files found in: {ksn_dir}")
        logger.error("Run calculate_ksn.py first.")
        sys.exit(1)
    
    # DEMs are in a parallel directory structure
    dems_dir = Path(r"E:\LiDAR\Scoped\Watersheds\DEMs")
    
    total = len(ksn_files)
    logger.info(f"Found {total} watershed ksn shapefiles")
    logger.info(f"Ksn dir    : {ksn_dir}")
    logger.info(f"DEMs dir   : {dems_dir}")
    logger.info(f"Output dir : {output_dir}")
    logger.info(f"Colormap   : {COLORMAP}")
    logger.info(f"DPI        : {DPI}")
    logger.info("-" * 60)
    
    start_time = time.time()
    succeeded  = 0
    failed     = 0
    skipped    = 0
    
    for i, ksn_shp in enumerate(ksn_files, start=1):
        # Derive corresponding DEM path
        watershed_id = ksn_shp.stem.replace('_ksn', '')
        dem_path     = dems_dir / f"{watershed_id}.tif"
        
        if not dem_path.exists():
            logger.error(f"[{i:3d}/{total}] FAIL  {watershed_id} — DEM not found: {dem_path}")
            failed += 1
            continue
        
        out_fig = output_dir / f"{watershed_id}_profile.png"
        if out_fig.exists():
            skipped += 1
            logger.info(f"[{i:3d}/{total}] SKIP  {watershed_id} — already exists")
            continue
        
        logger.info(f"[{i:3d}/{total}] START {watershed_id}")
        tile_start = time.time()
        
        success, result = plot_stream_profile(
            ksn_shp, dem_path, output_dir, logger
        )
        
        if success:
            succeeded += 1
            tile_time = time.time() - tile_start
            elapsed   = time.time() - start_time
            rate      = i / elapsed
            eta_min   = (total - i) / rate / 60 if rate > 0 else 0
            
            logger.info(
                f"[{i:3d}/{total}] OK    {watershed_id}  |  "
                f"{tile_time:.1f}s  |  "
                f"ETA {eta_min:.1f} min"
            )
        else:
            failed += 1
            logger.error(f"[{i:3d}/{total}] FAIL  {watershed_id}")
    
    elapsed_total = time.time() - start_time
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info(f"  Total watersheds : {total}")
    logger.info(f"  Succeeded        : {succeeded}")
    logger.info(f"  Skipped          : {skipped}")
    logger.info(f"  Failed           : {failed}")
    logger.info(f"  Output dir       : {output_dir}")
    logger.info(f"  Total time       : {elapsed_total / 60:.1f} minutes")


if __name__ == "__main__":
    main()
