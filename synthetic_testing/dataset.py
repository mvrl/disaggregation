from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
import os
import albumentations as A
from torchvision import transforms
import random
import torchvision
from skimage import io
from sklearn.model_selection import train_test_split
import torch

class Eurosat(torch.utils.data.Dataset):
    def __init__(self, mode='train', root='/localdisk0/SCRATCH/watch/EuroSAT/ds/images/remote_sensing/otherDatasets/sentinel_2/tif'):
        data = torchvision.datasets.DatasetFolder(root=root, loader=self.im_loader, transform=None, extensions='tif')

        if mode == 'train':
            train_set, _ = train_test_split(data, test_size=0.1, stratify=data.targets, random_state=42)
            self.dset = train_set
            self.transform = A.Compose([
              A.HorizontalFlip(p=0.5),
              A.RandomRotate90(p=1.0),
              A.NoOp()
            ],
            )
        elif mode == 'test' or mode == 'validation':
            _, val_set = train_test_split(data, test_size=0.1, stratify=data.targets, random_state=42)
            self.dset = val_set
            self.transform = A.Compose([
               A.NoOp()
            ],
            )

        self.rgb_mean = torch.tensor([ 946.2784, 1041.8102, 1117.2845]).unsqueeze(1).unsqueeze(1)
        self.rgb_std = torch.tensor([594.3585, 395.1237, 333.4564]).unsqueeze(1).unsqueeze(1)
        self.ir_mean = torch.tensor(2299.8538)
        self.ir_std = torch.tensor(1117.7264)
            
    def im_loader(self, path):
        image = np.asarray((io.imread(path)), dtype='float32')
        return image

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        item = self.dset[idx]

        transformed = self.transform(image=item[0])
        image = transformed['image']
        image = torch.tensor(image).permute(2, 0, 1)
        
        label = torch.tensor(item[0]).permute(2,0,1)
        label = image[7,:,:]
        image = image[[3,2,1],:,:]

        image = (image - self.rgb_mean) / self.rgb_std
        label = (label - self.ir_mean) / self.ir_std
        
        return {'image': image, 'label': label}

