import rasterio
with rasterio.open(r"E:\ksn-project\data\watersheds\watershed_1_fdr.tif") as src:
    print(src.nodata, src.dtype, src.crs)