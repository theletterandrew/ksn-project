import rasterio

fdr  = r"E:\ksn-project\data\watersheds\watershed_1_fdr.tif"
mask = r"E:\ksn-project\data\scratch\watersheds\_ws_mask_1.tif"

with rasterio.open(fdr) as s:
    print("FDR  :", s.width, s.height, s.transform, s.crs, s.nodata, s.dtypes)

with rasterio.open(mask) as s:
    print("MASK :", s.width, s.height, s.transform, s.crs, s.nodata, s.dtypes)


with rasterio.open(r"E:\ksn-project\data\watersheds\watershed_1_fdr.tif") as src:
    arr = src.read(1)
    nd  = src.nodata

valid = arr[arr != nd]
print("Unique FDR values:", np.unique(valid))

import subprocess
result = subprocess.run([r"E:\ksn-project\bin\WBT\whitebox_tools.exe", "--version"], capture_output=True, text=True)
print(result.stdout)