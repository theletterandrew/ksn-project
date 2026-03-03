import rasterio
with rasterio.open(r"E:\ksn-project\data\watersheds\watershed_1_fdr.tif") as src:
    print(src.nodata, src.dtypes, src.crs)