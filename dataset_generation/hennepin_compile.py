import os
from pickletools import uint8 
import pandas as pd
import numpy as np 
import geopandas as gpd
from tqdm import tqdm
import pickle
import random

import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_function
from torch.utils.data import Dataset

from PIL import Image

def build_dataset(df, data_dir, gdf, dest_folder, combine):
    all_images = []
    all_masks = []
    all_values = []
    all_rows = []
    all_pids = []
    
    img_dir_path = os.path.join(dest_folder, 'imgs')
    mask_dir_path = os.path.join(dest_folder, 'masks')
    os.makedirs(img_dir_path, exist_ok=True)
    os.makedirs(mask_dir_path, exist_ok=True)
    pkl_path = os.path.join(dest_folder, 'vals.pkl')

    for index,row in tqdm(df.iterrows(), total =len(df)):
        # Grab path from CSV dataframe
        dir_path = os.path.join(data_dir, str(int(row['lat_mid'])), str(int(row['lon_mid'])))

        # Load in masks, build aggregation matrix
        masks_dir = os.path.join(dir_path, 'masks')
        values = np.zeros(100, dtype=np.float64)
        pids = []
        masks = np.zeros((100,302,302), dtype=np.int8)
        
        leng = len(os.listdir(masks_dir))

        #segment out ones of this size
        if(leng <10 or leng > 100):
            continue

        #image
        image_name = os.path.join(dir_path, str(int(row['lat_mid']))+'_'+str(int(row['lon_mid']))+'.png')
        image = Image.open(image_name)
        #image = np.array(image)
        all_images.append(image)

        #all parcels
        if(os.path.isdir(masks_dir)):
            for index,filename in enumerate(os.listdir(masks_dir)):
                if(filename.endswith('.tif')):

                    #   Grab the PID filename
                    pid = os.path.splitext(filename)[0]
                    pids.append(pid)

                    # get parcel
                    parcel = gdf.loc[gdf['PID'] == pid]
                        # grab the value from the gdf
                    if(len(parcel['TOTAL_MV1'].values) == 1):
                        value = parcel['TOTAL_MV1'].values.item()
                        #print(value)
                    else:
                        #print("skipping a thing")
                        continue

                    values[index] = value
                    mask_img_path = os.path.join(masks_dir, filename)
                    
                    mask = Image.open(mask_img_path) # now we grab each mask
                    
                    mask = transforms_function.vflip(mask) #each mask is generated upside down
                    masks[index] = np.array(mask)

        if combine:
            geoms = []
            geom_index = []

            for i,pid in enumerate(pids):
                geoms.append( gdf.loc[gdf['PID'] == pid]['geometry'].values[0])
                geom_index.append(i)
                
            combine_length = int(len(geoms) / 2)

            for x in range(combine_length):
                #randomly select index
                randomINDEX = random.randint(0, len(geoms) - 1)
                parcel_geom = geoms[randomINDEX]
                
                #copy and remove the one we are looking at
                others = geoms.copy()
                del others[randomINDEX]
                others = gpd.GeoSeries(others)

                #Get the object with min distance
                distances = others.distance(parcel_geom)
                min_value = np.min( distances )
                
                #get the relative index
                closestGEOM = others[distances == min_value].values[0]
                closestINDEX = geoms.index(closestGEOM)
                
                #get the true indexes in the large datastructures
                closestINDEX_true = geom_index[closestINDEX]
                randomINDEX_true = geom_index[randomINDEX]

                mask_sum = masks[closestINDEX_true] + masks[randomINDEX_true]
                val_sum = values[closestINDEX_true] + values[randomINDEX_true]
                
                #Avoid outliers
                if(val_sum / np.count_nonzero(mask_sum) < 1000):
                    values[closestINDEX_true] = val_sum
                    masks[closestINDEX_true] = mask_sum > 0

                    #delete old
                    del geoms[randomINDEX]
                    del geom_index[randomINDEX]
                    masks[randomINDEX_true] = masks[randomINDEX_true] * 0
                    #masks = np.delete(masks,randomINDEX)
                    values[randomINDEX_true] = 0


        all_values.append(values)
        all_rows.append(row)
        all_pids.append(pids)
        all_masks.append(masks)

    for i,image in tqdm(enumerate(all_images)):
        img_save_path = os.path.join(dest_folder, 'imgs', str(i)+".png")
        image.save(img_save_path)
        mask_save_path = os.path.join(dest_folder, 'masks', str(i)+".pkl")
        with open(mask_save_path, 'wb') as f:
                pickle.dump(all_masks[i], f)

    with open(pkl_path, 'wb') as f:
                pickle.dump(all_values, f)


if __name__ == "__main__":
    random.seed(0)

    data_dir = '/u/eag-d1/data/Hennepin/new_area_302_fixed/'

    csv_path = os.path.join(data_dir, 'hennepin_bbox.csv')
    #shp_path = os.path.join(data_dir, 'hennepin.shp')
    shp_path = "/u/eag-d1/data/Hennepin/hennepin_county_parcels/hennepin_county_parcels.shp"
    ds_path = os.path.join(data_dir, 'dataset_compiled')

    df = pd.read_csv(csv_path)
    print("dataset_hennepin: Reading GeoDataFrame...")
    gdf = gpd.read_file(shp_path)
    print("dataset_hennepin: Done...")
    # Watch for outliers
    #gdf = gdf[gdf['TOTAL_MV1'].between(gdf['TOTAL_MV1'].quantile(0.05), gdf['TOTAL_MV1'].quantile(0.95))]
    print("dataset_hennepin: Reading Images, pickling...")

    path = '/u/eag-d1/data/Hennepin/compiled_302x302_gsd1_COMBINED_fixed/'
    build_dataset(df,data_dir,gdf,path, True)   