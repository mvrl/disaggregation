import pandas as pd
import os
from tqdm import tqdm
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import geopandas as gpd
import argparse
import shapely
from geofeather import to_geofeather, from_geofeather

# Code modifed from here
#https://gis.stackexchange.com/questions/352495/converted-vector-to-raster-file-is-black-and-white-in-colour-gdal-rasterize

parser = argparse.ArgumentParser(description='Download Image Set to folder')
parser.add_argument('--dir', type=str, help='Directory to download to.', default= './downloads/')
parser.add_argument('--gsd', type=int, help='Ground Sample Distance', default=1)
args = parser.parse_args()

# Local paths
parcels_file = './hennepin_county_parcels/hennepin_county_parcels.shp'

buildings_file = './hennepin_county_parcels/Minnesota_ESPG26915.shp'

# Rasterizing parcel masks
def raster_parcel_mask(bbox, row_bbox, fn, gdf):
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
def raster_buildings(bbox, row_bbox, fn, gdf):

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
def raster_boundary(bbox, row_bbox, fn, gdf):

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

# Rasterizing building masks 
def raster_masks(polygon, bbox, gdf, dir, combine_nearest = False):

    indexes = []

    if combine_nearest:

        for index,row in gdf.iterrows():
            region_polygon = row['geometry']
            gdf.drop(index)
            nearest_index = gdf.sindex.nearest(region_polygon)
            other_row = gdf.drop(nearest_index)
            other_row.append(row)
            print(other_row)
            


    for index,row in gdf.iterrows():

        region_polygon = row['geometry']

        fn = os.path.join(dir, str(row['PID']) + '.tif')

        if(region_polygon.within(polygon) and row['TOTAL_MV1'] > 0):

            indexes.append(index)

            #making the shapefile as an object.
            input_shp = ogr.Open(gpd.GeoSeries([region_polygon]).to_json())

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

    return indexes


# Loop through generated CSV
if __name__ == "__main__":

    shp_path = parcels_file

    csv_path = args.dir + 'hennepin_bbox.csv'

    new_shp_path = args.dir + 'hennepin.shp'

    feather_path = args.dir    
    bbox_df = pd.read_csv(csv_path)

    print("using Geofeather")
    feather_file = os.path.join(feather_path, os.path.basename(shp_path).replace('.shp', '.feather'))
    if os.path.exists(feather_file):
        gdf = from_geofeather(feather_file)
    else:
        gdf = gpd.read_file(shp_path)
        to_geofeather(gdf, feather_file)
    #shp_path = shp_path

    print( "done")

    gdf.set_crs("EPSG:26915")
    
    useful_indexes = []

    print("Rasterizing each label...")
    with tqdm(total = len(bbox_df)) as pbar:
        for index,row in bbox_df.iterrows():

            label_path = os.path.join(args.dir, str(int(row['lat_mid'])), str(int(row['lon_mid'])))

            parcel_value_path = os.path.join(label_path, 'parcel_value.tif')# need a better label filename 
            building_path = os.path.join(label_path, 'building_mask.tif')
            parcel_mask_path = os.path.join(label_path, 'parcel_mask.tif')
            boundary_path = os.path.join(label_path, 'parcel_boundary.tif')
            mask_dir = os.path.join(label_path, 'masks')

            #BBOXES with different orientations
            image_bbox = (row['lat_min'], row['lat_max'],row['lon_min'], row['lon_max'])
            row_bbox = (row['lat_min'], row['lon_min'],row['lat_max'], row['lon_max'])

            #polygon bounding box
            polygon = shapely.geometry.box(row['lat_min'], row['lon_min'], row['lat_max'], row['lon_max'])

            spatial_index = gdf.sindex
            possible_matches_index = list(spatial_index.intersection(polygon.bounds))
            possible_matches = gdf.iloc[possible_matches_index]
            precise_matches = possible_matches[possible_matches.intersects(polygon)]
            #print(precise_matches)

            # For each BBOX generate raster and filepath
            if not os.path.exists(parcel_mask_path):
                raster_parcel_mask(bbox = image_bbox,row_bbox= row_bbox,fn=parcel_mask_path, gdf = precise_matches)
            if not os.path.exists(building_path):
                raster_buildings(bbox = image_bbox,row_bbox= row_bbox,fn=building_path, gdf = precise_matches)
            if not os.path.exists(boundary_path):
                raster_boundary(bbox = image_bbox,row_bbox= row_bbox,fn=boundary_path, gdf = precise_matches)
            if not os.path.exists(mask_dir):
                os.mkdir(mask_dir)
            
            indexes = raster_masks(polygon, image_bbox, precise_matches, mask_dir, combine_nearest=True)
            #print(indexes)

            #useful_indexes.extend(indexes)
            #print(useful_indexes)


            pbar.update(1)

    #filtered_gdf = gdf.iloc[useful_indexes]

    #print(filtered_gdf)

    #filtered_gdf.to_file(new_shp_path)