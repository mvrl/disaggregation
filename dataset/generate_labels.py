import pandas as pd
import os
from tqdm import tqdm
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import geopandas as gpd
import argparse

# Code modifed from here
#https://gis.stackexchange.com/questions/352495/converted-vector-to-raster-file-is-black-and-white-in-colour-gdal-rasterize

parser = argparse.ArgumentParser(description='Download Image Set to folder')
parser.add_argument('--dir', type=str, help='Directory to download to.', default= './downloads/')
parser.add_argument('--gsd', type=int, help='Ground Sample Distance', default=1)
args = parser.parse_args()

# Local paths
parcels_file = './hennepin_county_parcels/hennepin_county_parcels.shp'

buildings_file = './hennepin_county_parcels/Minnesota_ESPG26915.shp'

# Rasterizing parcel values
def raster_parcel_values(bbox, row_bbox, fn):

    # Filter shapefile
    gdf = gpd.read_file(parcels_file, bbox = row_bbox)
    gdf.crs = "EPSG:26915"

    # Add Average Value
    gdf['AVERAGE_MV1'] = gdf['TOTAL_MV1'] / gdf['geometry'].area

    #making the shapefile as an object.
    input_shp = ogr.Open(gdf.to_json())

    #getting layer information of shapefile.
    shp_layer = input_shp.GetLayer()

    #pixel_size determines the size of the new raster.
    #pixel_size is proportional to size of shapefile.
    pixel_size = int(args.gsd)

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

# Rasterizing parcel values
def raster_parcel_mask(bbox, row_bbox, fn):

    # Filter shapefile
    gdf = gpd.read_file(parcels_file, bbox = row_bbox)
    gdf.crs = "EPSG:26915"

    # Add Average Value
    gdf['AVERAGE_MV1'] = gdf['TOTAL_MV1'] / gdf['geometry'].area

    #making the shapefile as an object.
    input_shp = ogr.Open(gdf.to_json())

    #getting layer information of shapefile.
    shp_layer = input_shp.GetLayer()

    #pixel_size determines the size of the new raster.
    #pixel_size is proportional to size of shapefile.
    pixel_size = int(args.gsd)

    #get extent values to set size of output raster.
    x_min, x_max, y_min, y_max = bbox

    #calculate size/resolution of the raster.
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)

    #get GeoTiff driver by 
    image_type = 'GTiff'
    driver = gdal.GetDriverByName(image_type)

    #passing the filename, x and y direction resolution, no. of bands, new raster.
    new_raster = driver.Create(fn, x_res, y_res, 1, gdal.GDT_Byte)

    #transforms between pixel raster space to projection coordinate space.
    new_raster.SetGeoTransform((x_min, pixel_size, 0, y_min, 0, pixel_size))

    #get required raster band.
    band = new_raster.GetRasterBand(1)

    #assign no data value to empty cells.
    no_data_value = 0
    band.SetNoDataValue(no_data_value)
    band.FlushCache()

    #main conversion method
    gdal.RasterizeLayer(new_raster, [1], shp_layer, burn_values=[1])

    #adding a spatial reference
    new_rasterSRS = osr.SpatialReference()
    new_rasterSRS.ImportFromEPSG(2975)
    new_raster.SetProjection(new_rasterSRS.ExportToWkt())

# Rasterizing building masks 
def raster_buildings(bbox, row_bbox, fn):

    gdf = gpd.read_file(buildings_file, bbox = row_bbox)

    gdf.crs = "EPSG:26915"

    #making the shapefile as an object.
    input_shp = ogr.Open(gdf.to_json())

    #getting layer information of shapefile.
    shp_layer = input_shp.GetLayer()

    #pixel_size determines the size of the new raster.
    #pixel_size is proportional to size of shapefile.
    pixel_size = int(args.gsd)

    #get extent values to set size of output raster.
    x_min, x_max, y_min, y_max = bbox

    #calculate size/resolution of the raster.
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)

    #get GeoTiff driver by 
    image_type = 'GTiff'
    driver = gdal.GetDriverByName(image_type)

    #passing the filename, x and y direction resolution, no. of bands, new raster.
    new_raster = driver.Create(fn, x_res, y_res, 1, gdal.GDT_Byte)

    #transforms between pixel raster space to projection coordinate space.
    new_raster.SetGeoTransform((x_min, pixel_size, 0, y_min, 0, pixel_size))

    #get required raster band.
    band = new_raster.GetRasterBand(1)

    #assign no data value to empty cells.
    no_data_value = 0
    band.SetNoDataValue(no_data_value)
    band.FlushCache()

    #main conversion method
    gdal.RasterizeLayer(new_raster, [1], shp_layer, burn_values=[1])

    #adding a spatial reference
    new_rasterSRS = osr.SpatialReference()
    new_rasterSRS.ImportFromEPSG(2975)
    new_raster.SetProjection(new_rasterSRS.ExportToWkt())

# Rasterizing building masks 
def raster_boundary(bbox, row_bbox, fn):

    gdf = gpd.read_file(parcels_file, bbox = row_bbox)

    gdf.crs = "EPSG:26915"

    gdf = gdf.boundary

    #making the shapefile as an object.
    input_shp = ogr.Open(gdf.to_json())

    #getting layer information of shapefile.
    shp_layer = input_shp.GetLayer()

    #pixel_size determines the size of the new raster.
    #pixel_size is proportional to size of shapefile.
    pixel_size = int(args.gsd)

    #get extent values to set size of output raster.
    x_min, x_max, y_min, y_max = bbox

    #calculate size/resolution of the raster.
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)

    #get GeoTiff driver by 
    image_type = 'GTiff'
    driver = gdal.GetDriverByName(image_type)

    #passing the filename, x and y direction resolution, no. of bands, new raster.
    new_raster = driver.Create(fn, x_res, y_res, 1, gdal.GDT_Byte)

    #transforms between pixel raster space to projection coordinate space.
    new_raster.SetGeoTransform((x_min, pixel_size, 0, y_min, 0, pixel_size))

    #get required raster band.
    band = new_raster.GetRasterBand(1)

    #assign no data value to empty cells.
    no_data_value = 0
    band.SetNoDataValue(no_data_value)
    band.FlushCache()

    #main conversion method
    gdal.RasterizeLayer(new_raster, [1], shp_layer, burn_values=[1])

    #adding a spatial reference
    new_rasterSRS = osr.SpatialReference()
    new_rasterSRS.ImportFromEPSG(2975)
    new_raster.SetProjection(new_rasterSRS.ExportToWkt())

# Loop through generated CSV
if __name__ == "__main__":

    csv_path = args.dir + 'hennepin_bbox.csv'

    bbox_df = pd.read_csv(csv_path)


    print("Rasterizing each label...")
    with tqdm(total = len(bbox_df)) as pbar:
        for index,row in bbox_df.iterrows():

            label_path = os.path.join(args.dir, str(int(row['lat_mid'])), str(int(row['lon_mid'])))

            parcel_value_path = os.path.join(label_path, 'parcel_value.tif')# need a better label filename 
            building_path = os.path.join(label_path, 'building_mask.tif')
            parcel_mask_path = os.path.join(label_path, 'parcel_mask.tif')
            boundary_path = os.path.join(label_path, 'parcel_boundary.tif')

            #BBOXES with different orientations
            image_bbox = (row['lat_min'], row['lat_max'],row['lon_min'], row['lon_max'])
            row_bbox = (row['lat_min'], row['lon_min'],row['lat_max'], row['lon_max'])

            # For each BBOX generate raster and filepath
            if not os.path.exists(parcel_value_path):
                raster_parcel_values(bbox = image_bbox,row_bbox= row_bbox,fn=parcel_value_path)
            if not os.path.exists(parcel_mask_path):
                raster_parcel_mask(bbox = image_bbox,row_bbox= row_bbox,fn=parcel_mask_path)
            if not os.path.exists(building_path):
                raster_buildings(bbox = image_bbox,row_bbox= row_bbox,fn=building_path)
            if not os.path.exists(boundary_path):
                raster_boundary(bbox = image_bbox,row_bbox= row_bbox,fn=boundary_path)

            pbar.update(1)