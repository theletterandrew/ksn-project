import rasterio
import numpy as np

for label, path in [
    ("FDR",        r"C:\Users\andre\OneDrive\Documents\ksn-project\data\scratch\WBT\flow_direction.tif"),
    ("Pour points", r"C:\Users\andre\OneDrive\Documents\ksn-project\data\scratch\watersheds\pourpoints_snapped.tif"),
]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        valid = arr[arr != src.nodata] if src.nodata is not None else arr.flatten()
        print(f"\n{label}")
        print(f"  CRS      : {src.crs}")
        print(f"  dtype    : {src.dtypes[0]}")
        print(f"  nodata   : {src.nodata}")
        print(f"  shape    : {src.shape}")
        print(f"  transform: {src.transform}")
        print(f"  valid cells: {len(valid)}")
        print(f"  unique values: {np.unique(valid)}")