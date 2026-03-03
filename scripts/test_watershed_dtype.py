import rasterio

fdr  = r"E:\ksn-project\data\watersheds\watershed_1_fdr.tif"
mask = r"E:\ksn-project\data\scratch\watersheds\_ws_mask_1.tif"

with rasterio.open(fdr) as s:
    print("FDR  :", s.width, s.height, s.transform, s.crs, s.nodata, s.dtypes)

with rasterio.open(mask) as s:
    print("MASK :", s.width, s.height, s.transform, s.crs, s.nodata, s.dtypes)