import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import os
import PIL
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_function
from torch.utils.data import Dataset, DataLoader
from config import cfg
import geopandas as gpd

class dataset_hennepin(Dataset):        # derived from 'dataset_SkyFinder_multi_clean', applies random crop
    def __init__(self, mode, data_dir, csv_path, shp_path):
        
        self.mode = mode
        self.data_dir = data_dir

        self.df = pd.read_csv(csv_path)

        print("Reading GeoDataFrame...")
        self.gdf = gpd.read_file(shp_path)
        self.shp_path = shp_path
        self.gdf['AVERAGE_MV1'] = self.gdf['TOTAL_MV1'] / self.gdf['geometry'].area
        print("Done")
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Grab path from CSV dataframe
        row = self.df.iloc[idx]
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
                        value = self.gdf.loc[ self.gdf['PID'] == pid ]['AVERAGE_MV1'].values

                        # now we grab each mask
                        mask = Image.open(img_path)

                        # flatten the image
                        mask = np.array(mask).flatten()

                        masks.append(mask)
                        values.append(value)

                parcel_masks = np.vstack(masks)
                parcel_values = np.vstack(values)
            else:
                parcel_masks = []
                parcel_values = []


        # image
        image_name = os.path.join(dir_path, str(int(row['lat_mid']))+'.0_'+str(int(row['lon_mid']))+'.0.tif')
        image = Image.open(image_name)

        # parcel value map
        parcel_fname = os.path.join(dir_path, 'parcel_value.tif')
        value_map = Image.open(parcel_fname)
        value_map = transforms_function.vflip(value_map)
        parcel_fname = os.path.join(dir_path, 'value.tif')


        row_bbox = (row['lat_min'], row['lon_min'],row['lat_max'], row['lon_max'])
        #Region Array
        shp_filename = os.path.join(dir_path, 'parcels.shp')
        parcel_ids = []
        if(os.path.exists(shp_filename)):
            chip_gdf = gpd.read_file(shp_filename, row_bbox)
            for PID in chip_gdf['PID']:
                parcel_ids.append(PID)
        
        if self.mode == 'train':            # random flips during training
            if random.random() > 0.5:
                image = transforms_function.hflip(image)
                value_map = transforms_function.hflip(value_map)
                
            if random.random() > 0.5:
                image = transforms_function.vflip(image)
                value_map = transforms_function.vflip(value_map)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256, 256))
        
        image = transforms_function.crop(image, i, j, h, w)
        value_map = transforms_function.crop(value_map, i, j, h, w)    

        image = self.to_tensor(image)
        # note: no ImageNet normalization applied yet
        
        value_map = torch.from_numpy(np.array(value_map))

        return image, value_map, parcel_masks, parcel_values


def get_data(cfg, mode, data_dir=cfg.data.root_dir):
    # expects a mode of dataloader
    # valid options: 'train', 'val', 'test'
    
    this_dataset = dataset_hennepin(mode=mode, data_dir=data_dir, csv_path = cfg.data.csv_path)

    torch.manual_seed(0)

    #Split Sizes
    train_size = int( np.floor( len(this_dataset) * (1-cfg.train.validation_split-cfg.train.test_split) ) )
    val_size = int( np.floor( len(this_dataset) * cfg.train.validation_split) )
    test_size = int( np.floor( len(this_dataset) * cfg.train.test_split ) )

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(this_dataset, [train_size, val_size, test_size])

    if(mode == 'train'):
        data_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=cfg.train.shuffle,
                             num_workers=cfg.train.num_workers)
    elif(mode == 'val'):
        data_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=cfg.train.shuffle,
                             num_workers=cfg.train.num_workers)
    elif(mode == 'test'):
        data_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=cfg.train.shuffle,
                             num_workers=cfg.train.num_workers)

    
    return data_loader

