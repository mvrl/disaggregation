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
    def __init__(self, mode, data_dir, csv_path):
        
        self.mode = mode
        self.data_dir = data_dir

        self.df = pd.read_csv(csv_path)

        #self.gdf = gpd.read_file(shp_path)
        #self.shp_path = shp_path
        #self.gdf['AVERAGE_MV1'] = self.gdf['TOTAL_MV1'] / self.gdf['geometry'].area
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet
        
        #read split file
        fname = os.path.join('split_files', mode+'.txt')
        with open(fname, 'r') as f:
            self.full_list = [i[:-1] for i in f]
        
        
    def __len__(self):
        return len(self.df)

            
    def generate_Aggregation_matrix(self):

        parcel_ids = []
        masks = []
        values = []

        for i in range(0,len(self)):
            print(i)
            
            row = self.df.iloc[i]
            dir_path = os.path.join(self.data_dir, str(int(row['lat_mid'])), str(int(row['lon_mid'])))
            
            masks_dir = os.path.join(dir_path, 'masks')

            if(os.path.isdir(masks_dir)):
                #Use os listdir to list each file in masks

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

                        # add to lists
                        parcel_ids.append(pid)
                        masks.append(mask)
                        values.append(value)


        #now make a dataframe 
        df = pd.DataFrame({'PID':parcel_ids, 'value': values})

        agg = np.vstack(masks)

        return df, agg


    # default here is uniform value
    def get_MASTER(self):
        return self.gdf['AVERAGE_MV1']

    def __getitem__(self, idx):
        # Grab path from CSV dataframe
        row = self.df.iloc[idx]
        dir_path = os.path.join(self.data_dir, str(int(row['lat_mid'])), str(int(row['lon_mid'])))

        # read building mask
        building_path = os.path.join(dir_path, 'building_mask.tif')
        building_mask =  Image.open(building_path)
        building_mask = transforms_function.vflip(building_mask) # original labels are vertically flipped

        #Read in Parcel Boundary
        parcel_boundary_path = os.path.join(dir_path, 'parcel_boundary.tif')
        parcel_boundary = Image.open(parcel_boundary_path)
        parcel_boundary = transforms_function.vflip(parcel_boundary)

        #Read in Parcel Mask
        parcel_mask_path = os.path.join(dir_path, 'parcel_mask.tif')
        parcel_mask = Image.open(parcel_mask_path)
        parcel_mask = transforms_function.vflip(parcel_mask)

        #Combine masks to multiclass label
        #addition_one = np.array(parcel_mask) + np.array(building_mask) * 2
        #multiclass_label = np.where(addition_one > 2, 2, addition_one)
        #addition_two  = overlap_fixed + np.array(building_mask) * 3
        #multiclass_label = np.where(addition_two > 3, 3, addition_two)
        
        multiclass_label = np.array(building_mask)
        #NEEDED
        multiclass_label = np.where(multiclass_label < 0, 0, multiclass_label)
        multiclass_label = Image.fromarray(multiclass_label)
        

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
            # i need to calculate the indexes here
            # select the indexes of the main gdf with matching IDs of the chips gdf
            for PID in chip_gdf['PID']:
                parcel_ids.append(PID)
            

        
        if self.mode == 'train':            # random flips during training
            if random.random() > 0.5:
                image = transforms_function.hflip(image)
                multiclass_label = transforms_function.hflip(multiclass_label)
                value_map = transforms_function.hflip(value_map)
                
            if random.random() > 0.5:
                image = transforms_function.vflip(image)
                multiclass_label = transforms_function.vflip(multiclass_label)
                value_map = transforms_function.vflip(value_map)
                
            # note: there is no random cropping yet



        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256, 256))
        
        image = transforms_function.crop(image, i, j, h, w)
        multiclass_label = transforms_function.crop(multiclass_label, i, j, h, w)
        value_map = transforms_function.crop(value_map, i, j, h, w)    

        image = self.to_tensor(image)
        # note: no ImageNet normalization applied yet
        
        multiclass_label = torch.from_numpy(np.array(multiclass_label))
        value_map = torch.from_numpy(np.array(value_map))

        return image, multiclass_label, value_map, parcel_ids


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

