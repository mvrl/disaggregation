from matplotlib import image
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

from sklearn.utils import check_random_state
from scipy.spatial.distance import cdist
import numpy as np

def prep_y_simple(X, centroids):
    """ Creates a simple target Y tensor, in this case, binary."""
    N,d,h,w = X.shape
    num_centroids = 1
    threshold = 0.02
    seed = 0
    rng = check_random_state(seed)
    color_vecs = X.permute(0,2,3,1).reshape(-1,3)
    source_colors = color_vecs[centroids,:]
    tmp = cdist(source_colors, color_vecs)
    tmp = np.min(tmp, axis=0) < threshold
    Y = (tmp).reshape(N,1,h,w).astype(np.float32)
    return Y
    
def get_centroids():
    """ Creates a simple target Y tensor, in this case, binary."""
    shape = 1024 # HARDCODED
    num_centroids = 1
    seed = 0
    rng = check_random_state(seed)
    centroids = rng.choice(shape,num_centroids)
    return centroids


class Eurosat(torch.utils.data.Dataset):
    def __init__(self, mode='train', root='/localdisk0/SCRATCH/EuroSAT/ds/images/remote_sensing/otherDatasets/sentinel_2/tif'):
        data = torchvision.datasets.DatasetFolder(root=root, loader=self.im_loader, transform=None, extensions='tif')

        self.centroids = get_centroids()

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

class Cifar(Dataset):
    def __init__(self, mode):
        self.totnsr = transforms.ToTensor()

        self.dataset = []

        self.mode = mode

        if(self.mode == 'train'):
            self.dataset = torchvision.datasets.CIFAR10(root = './', train=True, download=True, transform=None)
            self.transform = A.Compose([
              A.HorizontalFlip(p=0.5),
              A.RandomRotate90(p=1.0),
              A.NoOp()
            ])
        elif mode == 'test':
            self.dataset = torchvision.datasets.CIFAR10(root = './', train=False, download=True, transform=None)
            test_set, val_set = train_test_split(self.dataset, test_size=0.5, stratify=self.dataset.targets, random_state=42)
            self.dataset = test_set
            self.transform = A.Compose([
              A.NoOp()
            ])
        elif mode == 'validation':
            self.dataset = torchvision.datasets.CIFAR10(root = './', train=False, download=True, transform=None)
            test_set, val_set= train_test_split(self.dataset, test_size=0.5, stratify=self.dataset.targets, random_state=42)
            self.dataset = val_set
            self.transform = A.Compose([
              A.NoOp()
            ])


        self.centroids = get_centroids()

        self.max_regions = 10

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        image_tensor = self.transform(image = np.array(sample[0]))
        image_tensor = self.totnsr(image_tensor['image'])

        Y = prep_y_simple(image_tensor.unsqueeze(0), self.centroids)
        Y = torch.tensor(Y).squeeze(0).squeeze(0)

        return {'image': image_tensor, 'label': Y}
