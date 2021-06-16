import torch
import numpy as np
import pandas as pd
import random
import os
import PIL
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_function
from torch.utils.data import Dataset
import geopandas as gpd

# OHHH CRAP I NEVER FLIPPED THE MASKS FOR PARCELS
# GOD I KNEW THAT WOULD BITE ME IN THE ASS
# I Wonder What kind of performance I can get with just that fixed

class dataset_hennepin(Dataset):        # derived from 'dataset_SkyFinder_multi_clean', applies random crop
    def __init__(self, mode, data_dir, csv_path, shp_path):
        
        self.mode = mode
        self.data_dir = data_dir

        self.df = pd.read_csv(csv_path)

        print("Reading GeoDataFrame...")
        self.gdf = gpd.read_file(shp_path)
        self.shp_path = shp_path

        self.gdf['AVERAGE_MV1'] = self.gdf['TOTAL_MV1'] / self.gdf['geometry'].area
        self.gdf = self.gdf[self.gdf['AVERAGE_MV1'].between(self.gdf['AVERAGE_MV1'].quantile(0.1), self.gdf['AVERAGE_MV1'].quantile(0.9))]
        #Normalize data
        self.gdf['AVERAGE_MV1'] = (self.gdf['AVERAGE_MV1'] - min( self.gdf['AVERAGE_MV1'] )) / ( max(self.gdf['AVERAGE_MV1']) - min(self.gdf['AVERAGE_MV1']))
        print("Done")

        print("Generating list of useful chips")
        self.rows = []
        for index,row in self.df.iterrows():
            dir_path = os.path.join(self.data_dir, str(int(row['lat_mid'])), str(int(row['lon_mid'])))
            masks_dir = os.path.join(dir_path, 'masks')
            if(os.path.isdir(masks_dir)):
                if os.listdir(masks_dir) != []:
                    self.rows.append(row)


        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet
        
    def __len__(self):
        return len(self.rows)

    def getgdf(self):
        return self.gdf

    def __getitem__(self, idx):
        # Grab path from CSV dataframe
        row = self.rows[idx]
        dir_path = os.path.join(self.data_dir, str(int(row['lat_mid'])), str(int(row['lon_mid'])))

        # Load in masks, build aggregation matrix
        masks_dir = os.path.join(dir_path, 'masks')

        masks = []
        values = []

        if(os.path.isdir(masks_dir)):
            if os.listdir(masks_dir) != []:
                for filename in os.listdir(masks_dir):
                    if(filename.endswith('.tif')):
                        # full file path
                        img_path = os.path.join(masks_dir, filename)

                        #   Grab the PID filename
                        pid = os.path.splitext(filename)[0]
                        
                        # grab the value from the gdf
                        value = self.gdf.loc[ self.gdf['PID'] == pid ]['AVERAGE_MV1'].values.item()

                        # now we grab each mask
                        mask = Image.open(img_path)

                        masks.append(mask)
                        values.append(value)
   
        # image
        image_name = os.path.join(dir_path, str(int(row['lat_mid']))+'.0_'+str(int(row['lon_mid']))+'.0.tif')
        image = Image.open(image_name)

        # parcel value map
        parcel_fname = os.path.join(dir_path, 'parcel_value.tif')
        value_map = Image.open(parcel_fname)
        value_map = transforms_function.vflip(value_map)
        parcel_fname = os.path.join(dir_path, 'value.tif')


        if self.mode == 'train':            # random flips during training
            if random.random() > 0.5:
                image = transforms_function.hflip(image)
                value_map = transforms_function.hflip(value_map)

                #Add loop for masks
                masks = [transforms_function.hflip(mask) for mask in masks]
                
            if random.random() > 0.5:
                image = transforms_function.vflip(image)
                value_map = transforms_function.vflip(value_map)

                #Add loop for masks
                masks = [transforms_function.vflip(mask) for mask in masks]

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256, 256))
        
        image = transforms_function.crop(image, i, j, h, w)
        value_map = transforms_function.crop(value_map, i, j, h, w)
        masks = [transforms_function.crop(mask, i, j, h, w) for mask in masks]

        image = self.to_tensor(image)
        # note: no ImageNet normalization applied yet

        #Prepare masks here
        
        #for each mask turn to numpy array, flatten, and vstack
        masks = [torch.from_numpy( np.array(mask).flatten()) for mask in masks]

        parcel_masks = np.vstack(masks)
        parcel_values = np.vstack(values)
        
        value_map = torch.from_numpy(np.array(value_map))

        parcel_values = torch.from_numpy(parcel_values)

        return image, parcel_masks, parcel_values