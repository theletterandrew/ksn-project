import fiona
with fiona.open(r"E:\ksn-project\data\scratch\watersheds\pourpoints_final.shp") as shp:
    print("Schema:", shp.schema)
    for feat in shp:
        print(feat["properties"])