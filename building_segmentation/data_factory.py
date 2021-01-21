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

class dataset_hennepin(Dataset):        # derived from 'dataset_SkyFinder_multi_clean', applies random crop
    def __init__(self, mode, data_dir, csv_path):
        
        self.mode = mode
        self.data_dir = data_dir

        self.csv = pd.read_csv(csv_path)
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet
        
        #read split file
        fname = os.path.join('split_files', mode+'.txt')
        with open(fname, 'r') as f:
            self.full_list = [i[:-1] for i in f]
        
        
    def __len__(self):
        return len(self.full_list)
    
    def __getitem__(self, idx):
        # Grab path from CSV
        row = self.csv.iloc[idx]
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
        addition_one = np.array(parcel_mask) + np.array(parcel_boundary) * 2
        overlap_fixed = np.where(addition_one > 2, 2, addition_one)
        addition_two  = overlap_fixed + np.array(building_mask) * 3
        multiclass_label = np.where(addition_two > 3, 3, addition_two)
        multiclass_label = np.where(multiclass_label < 0, 0, multiclass_label)
        multiclass_label = Image.fromarray(multiclass_label)

        # image
        image_name = os.path.join(dir_path, str(int(row['lat_mid']))+'.0_'+str(int(row['lon_mid']))+'.0.tif')
        image = Image.open(image_name)

        # parcel value
        parcel_fname = os.path.join(dir_path, 'parcel_value.tif')
        value = Image.open(parcel_fname)
        
        if self.mode == 'train':            # random flips during training
            if random.random() > 0.5:
                image = transforms_function.hflip(image)
                multiclass_label = transforms_function.hflip(multiclass_label)
                value = transforms_function.hflip(value)
                
            if random.random() > 0.5:
                image = transforms_function.vflip(image)
                multiclass_label = transforms_function.vflip(multiclass_label)
                value = transforms_function.vflip(value)
                
            # note: there is no random cropping yet


        image = self.to_tensor(image)
        # note: no ImageNet normalization applied yet
        
        multiclass_label = torch.from_numpy(np.array(multiclass_label))
        value = torch.from_numpy(np.array(value))


        return image, multiclass_label, value


def get_data(cfg, mode, data_dir=cfg.data.root_dir):
    # expects a mode of dataloader
    # valid options: 'train', 'val', 'test'
    
    this_dataset = dataset_hennepin(mode=mode, data_dir=data_dir, csv_path = cfg.data.csv_path)
        
    data_loader = DataLoader(this_dataset, batch_size=cfg.train.batch_size, shuffle=cfg.train.shuffle,
                             num_workers=cfg.train.num_workers)
    
    return data_loader

