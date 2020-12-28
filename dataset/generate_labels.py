import pandas as pd
import os
from tqdm import tqdm
import gdal
from osgeo import osr
from osgeo import ogr
import geopandas as gpd

# Code modifed from here
#https://gis.stackexchange.com/questions/352495/converted-vector-to-raster-file-is-black-and-white-in-colour-gdal-rasterize

#this needs a full path
shapefile = './hennepin_county_parcels/hennepin_county_parcels.shp'

csv_path = './hennepin_bbox.csv'

root_dir = './image_set'


def raster(bbox, row_bbox, fn):

    gdf = gpd.read_file(shapefile, bbox = row_bbox)
    gdf.crs = "EPSG:26915"

    gdf['AVERAGE_MV1'] = gdf['TOTAL_MV1'] / gdf['geometry'].area
    #print(gdf['AVERAGE_MV1'])
    #gdf.to_file(shapefile)

    #making the shapefile as an object.
    input_shp = ogr.Open(gdf.to_json())

    #getting layer information of shapefile.
    shp_layer = input_shp.GetLayer()

    #pixel_size determines the size of the new raster.
    #pixel_size is proportional to size of shapefile.
    pixel_size = 2

    #get extent values to set size of output raster.
    x_min, x_max, y_min, y_max = bbox

    #calculate size/resolution of the raster.
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)

    #get GeoTiff driver by 
    image_type = 'GTiff'
    driver = gdal.GetDriverByName(image_type)

    #passing the filename, x and y direction resolution, no. of bands, new raster.
    new_raster = driver.Create(fn, x_res, y_res, 1, gdal.GDT_Int32)

    #transforms between pixel raster space to projection coordinate space.
    new_raster.SetGeoTransform((x_min, pixel_size, 0, y_min, 0, pixel_size))

    #get required raster band.
    band = new_raster.GetRasterBand(1)

    #assign no data value to empty cells.
    no_data_value = 0
    band.SetNoDataValue(no_data_value)
    band.FlushCache()

    #main conversion method
    gdal.RasterizeLayer(new_raster, [1], shp_layer, options=['ATTRIBUTE=AVERAGE_MV1'])

    #adding a spatial reference
    new_rasterSRS = osr.SpatialReference()
    new_rasterSRS.ImportFromEPSG(2975)
    new_raster.SetProjection(new_rasterSRS.ExportToWkt())

if __name__ == "__main__":
    bbox_df = pd.read_csv(csv_path)

    # Loop through CSV

    print("Rasterizing each label...")
    with tqdm(total = len(bbox_df)) as pbar:
        for index,row in bbox_df.iterrows():

            label_path = os.path.join(root_dir, str(int(row['lat_mid'])), str(int(row['lon_mid'])))

            label_path = os.path.join(label_path, 'label.tif') # need a better label filename 

            image_bbox = (row['lat_min'], row['lat_max'],row['lon_min'], row['lon_max'])
            row_bbox = (row['lat_min'], row['lon_min'],row['lat_max'], row['lon_max'])

            #print(label_path)


            # For each BBOX generate raster and filepath
            raster(bbox = image_bbox,row_bbox= row_bbox,fn= label_path)

            pbar.update(1)