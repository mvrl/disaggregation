import os
import torch
import random
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms.functional as transforms_function
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class HennepinStyleTransfer(Dataset):
    def __init__(self, root_dirs, mode, crop_size, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dirs (list) List of directories (strings)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #Find associated csv, assume data matches for root_dirs given
        self.csv_path = os.path.join(root_dirs[0], 'hennepin_bbox.csv')
        self.df = pd.read_csv(self.csv_path)

        self.root_dirs = root_dirs
        self.mode = mode
        self.transform = transform
        self.crop_size = crop_size

    def __len__(self):
        return len(self.bbox_frame)

    def __getitem__(self, idx):
        # Grab bbox from from csv
        row = self.df.iloc[idx]
        img1_dir = os.path.join(self.root_dirs[0], str(int(row['lat_mid'])), str(int(row['lon_mid'])))
        img2_dir = os.path.join(self.root_dirs[1], str(int(row['lat_mid'])), str(int(row['lon_mid'])))

        #open images
        img1_path = os.path.join(img1_dir, str(int(row['lat_mid']))+'.0_'+str(int(row['lon_mid']))+'.0.png')
        img2_path = os.path.join(img2_dir, str(int(row['lat_mid']))+'.0_'+str(int(row['lon_mid']))+'.0.png')
        image_1 = Image.open(img1_path)
        image_2 = Image.open(img2_path)
        
        # Two different Random crops
        i, j, h, w = transforms.RandomCrop.get_params(image_1, output_size=self.crop_size)
        a, b, c, d = transforms.RandomCrop.get_params(image_1, output_size=self.crop_size)

        # TODO: check if tuple parameters are the same...

        # do the same crop on both, then do seperate crop on #1
        image1_aligned = transforms_function.crop(image_1, i, j, h, w)
        image2_aligned = transforms_function.crop(image_2, i, j, h, w)
        image1_unaligned = transforms_function.crop(image_1, a, b, c, d)

        if self.mode == 'train':            # random flips during training
            if random.random() > 0.5:
                image1_aligned = transforms_function.hflip(image1_aligned)
                image2_aligned = transforms_function.hflip(image2_aligned)
                image1_unaligned = transforms_function.hflip(image1_unaligned)
                
            if random.random() > 0.5:
                image1_aligned = transforms_function.vflip(image1_aligned)
                image2_aligned = transforms_function.vflip(image2_aligned)
                image1_unaligned = transforms_function.vflip(image1_unaligned)

        sample = {'image1_aligned':image1_aligned, 'image2_aligned': image2_aligned, 'image1_unaligned': image1_unaligned}

        if self.transform:
            sample['image1_aligned'] = self.transform(sample['image1_aligned'])
            sample['image2_aligned'] = self.transform(sample['image2_aligned'])
            sample['image1_unaligned'] = self.transform(sample['image1_unaligned'])

        return sample

if __name__ == "__main__":

    dataset = HennepinStyleTransfer(['/u/eag-d1/data/Hennepin/2020','/u/eag-d1/data/Hennepin/2018'],'train', crop_size = (100,100),
                            transform = transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),       
                            ]))

    print(dataset[0])