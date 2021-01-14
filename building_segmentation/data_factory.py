import torch
import torch.nn as nn
import numpy as np
import random
import os
import PIL
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_function
from torch.utils.data import Dataset, DataLoader
from config import cfg

class dataset_hennepin(Dataset):        # derived from 'dataset_SkyFinder_multi_clean', applies random crop
    def __init__(self, mode, data_dir):
        
        self.mode = mode
        self.data_dir = data_dir
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet
        
        # read split file
        fname = os.path.join('split_files', mode+'.txt')
        with open(fname, 'r') as f:
            self.full_list = [i[:-1] for i in f]
        
        
    def __len__(self):
        return len(self.full_list)
    
    def __getitem__(self, idx):
        # read building mask
        building_name = self.full_list[idx]
        building_fname = os.path.join(self.data_dir, building_name)
        building_mask =  Image.open(building_fname)
        building_mask = transforms_function.vflip(building_mask) # original labels are vertically flipped

        # image
        name_split = building_name.split('/')
        subdir_name = '/'.join(name_split[:-1])
        image_name = os.path.join(self.data_dir, subdir_name, name_split[1]+'.0_'+name_split[2]+'.0.tif')
        image = Image.open(image_name)

        # parcel value
        parcel_fname = os.path.join(self.data_dir, subdir_name, 'parcel_value.tif')
        value = Image.open(parcel_fname)
        
        if self.mode == 'train':
            # random flips during training
            if random.random() > 0.5:
                image = transforms_function.hflip(image)
                building_mask = transforms_function.hflip(building_mask)
                value = transforms_function.hflip(value)
                
            if random.random() > 0.5:
                image = transforms_function.vflip(image)
                building_mask = transforms_function.vflip(building_mask)
                value = transforms_function.vflip(value)
                
            # note: there is no random cropping yet


        image = self.to_tensor(image)
        # note: no ImageNet normalization applied yet
        
        
        building_mask = torch.from_numpy(np.array(building_mask))
        value = torch.from_numpy(np.array(value))


        return image, building_mask, value


def get_data(cfg, mode, data_dir=cfg.data.root_dir):
    # expects a mode of dataloader
    # valid options: 'train', 'val', 'test'
    
    this_dataset = dataset_hennepin(mode=mode, data_dir=data_dir)
        
    data_loader = DataLoader(this_dataset, batch_size=cfg.train.batch_size, shuffle=cfg.train.shuffle,
                             num_workers=cfg.train.num_workers)
    
    return data_loader
