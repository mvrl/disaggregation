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
import torch.nn as nn
import torch.distributions as dist

class Eurosat(torch.utils.data.Dataset):
   
    def __init__(self, mode='train', root='/u/eag-d1/data/Hennepin/EuroSAT/ds/images/remote_sensing/otherDatasets/sentinel_2/tif'):
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

        # No more NIR
        #self.ir_mean = torch.tensor(2299.8538)
        #self.ir_std = torch.tensor(1117.7264)


    def im_loader(self, path):

        image = np.asarray((io.imread(path)), dtype='float32')
       # image = (image - image.min()) / (image.max()-image.min())
        return image

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx, print_std=False):
        item = self.dset.__getitem__(idx)

        # EuroSAT MS data
        transformed = self.transform(image=item[0])
        image = transformed['image']
        image = torch.tensor(image).permute(2, 0, 1)

        # Image and Normalize
        image = image[[3,2,1],:,:]
        image = (image - self.rgb_mean) / self.rgb_std

        # Generate Some Model distribution
        # mu: (r+g)/2
        # std: 2b-1
        mu_true = (image[0,:,:] + image[1,:,:] ) / 2 
        std_true = torch.nn.functional.softplus(2*image[2,:,:] - 1)

        if print_std:
          print(f"true std: {std_true.mean()}")
      
        # Sample generated model for final label
        label_dist = torch.distributions.Normal(mu_true, std_true)
        label = label_dist.sample() 
        
        return {'image': image, 'label': label}


# This doesnt seem to work.
class dataset_cifar(Dataset):
    def __init__(self, mode):
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.dataset = []
        self.mode = mode

        if(self.mode == 'train'):
            self.dataset = torchvision.datasets.CIFAR10(root = './', train=True, download=True, transform=None)
        else:
            self.dataset = torchvision.datasets.CIFAR10(root = './', train=False, download=True, transform=None)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx, print_std=False):
        sample = self.dataset[idx]
        image = self.transform(sample[0])

        # this was the generative model Cohen defined
        #mu_true = 4*image[0,:,:] - 2*image[1,:,:]

        # here's one that Nathan defined, making it a little harder
        mu_true = image[0,:,:].clone()
        image[0,:,:] = 0.5
        
        std_true = nn.functional.softplus(.1*image[2,:,:] - 3)

        if print_std:
          print(f"true std: {std_true.mean()}")
        label_dist = dist.Normal(mu_true, std_true)
        label = label_dist.sample()

        return {'image': image, 'label': label}