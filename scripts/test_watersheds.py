import rasterio
import numpy as np

dem_path = r"E:\ksn-project\data\watersheds\watershed_1.tif"
fac_path = r"E:\ksn-project\data\watersheds\watershed_1_fac.tif"

with rasterio.open(dem_path) as src:
    dem    = src.read(1)
    nodata = src.nodata
    print(f"DEM nodata value : {nodata}")
    print(f"DEM unique values at corners: {dem[0,0]}, {dem[0,-1]}, {dem[-1,0]}, {dem[-1,-1]}")
    if nodata is not None:
        print(f"DEM nodata cell count: {(dem == nodata).sum():,}")
    else:
        print("DEM has no nodata defined")

with rasterio.open(fac_path) as src:
    fac    = src.read(1)
    nodata = src.nodata
    print(f"FAC nodata value : {nodata}")
    print(f"FAC max          : {fac[fac != nodata].max() if nodata is not None else fac.max():,.0f}")
    print(f"FAC cells > 0    : {(fac > 0).sum():,}")
    if nodata is not None:
        print(f"FAC nodata cells : {(fac == nodata).sum():,}")