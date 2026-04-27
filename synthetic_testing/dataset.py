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
   
    
    def __init__(self, mode='train', root=None):
        root = root or os.environ.get('EUROSAT_ROOT')
        if not root:
            raise ValueError(
                "Dataset root must be provided via the `root` argument or the `EUROSAT_ROOT` environment variable."
            )
        root = os.path.abspath(root)
        data = torchvision.datasets.DatasetFolder(root=root, loader=self.im_loader, transform=None, extensions=('.tif',))

        if mode == 'train':
            train_set, _ = train_test_split(data, test_size=0.1, stratify=data.targets, random_state=42)
            self.dset = train_set
            self.transform = A.Compose([
              A.HorizontalFlip(p=0.5),
            ],
            )
        elif mode == 'test' or mode == 'validation':
            _, val_set = train_test_split(data, test_size=0.1, stratify=data.targets, random_state=42)
            self.dset = val_set
            self.transform = A.Compose([
               A.NoOp()
            ],
            )

    def im_loader(self, path):
   
        image = np.asarray((io.imread(path)), dtype='float32')
       # image = (image - image.min()) / (image.max()-image.min())
        return image

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        item = self.dset.__getitem__(idx)

        transformed = self.transform(image=item[0])
        image = transformed['image']
        image = torch.tensor(image).permute(2, 0, 1)

        #label = torch.tensor(item[0]).permute(2,0,1)
        label = image[7,:,:]
        image = image[[3,2,1],:,:]
        
        label = (label - torch.mean(label)) / torch.std(label)
       # label = (label - torch.min(label)) / (torch.max(label)-torch.min(label))
        
        channel_mean = image.mean(dim=(1, 2), keepdim=True)
        channel_std = image.std(dim=(1, 2), keepdim=True)
        image = (image - channel_mean) / channel_std
        return {'image': image, 'label': label}

