import fiona
import geopandas as gpd

pour  = gpd.read_file(r"E:\ksn-project\data\scratch\watersheds\pourpoints_final.shp")
wshed = gpd.read_file(r"E:\ksn-project\data\scratch\watersheds\watersheds.shp")

print("Pour points CRS:", pour.crs)
print("Watersheds CRS:", wshed.crs)

joined = gpd.sjoin(pour, wshed[["gridcode", "geometry"]], how="left", predicate="within")
print(joined[["POUR_ID", "gridcode", "geometry"]])