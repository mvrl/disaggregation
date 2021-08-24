import torch
import torchvision.utils as t_util
import torch.nn as nn
import numpy as np
from torch.utils.data.dataloader import DataLoader
from config import cfg
import data_factory
import matplotlib.pyplot as plt

# CIFAR code from original paper
# https://github.com/orbitalinsight/region-aggregation-public/blob/master/run_cifar10.py

''' 
    Losses
'''
class regionAgg_layer(nn.Module):

    def __init__(self):
        super(regionAgg_layer, self).__init__()

    def forward(self, x, parcel_mask_batch):
        arr = []

        for i, item in enumerate(parcel_mask_batch):

            # x[i] is per pixel value estimation, flattened
            # item is a matrix of flattened parcel masks

            arr.append(torch.matmul(x[i], torch.from_numpy(item).T.float().to('cuda')))

        return arr
def MSE(outputs, targets):
    losses = []

    for output,target in zip(outputs,targets):
        losses.append( torch.sum((output - target)**2) ) 

    return torch.stack(losses, dim=0).mean()
def MAE(outputs, targets):
    losses = []

    for output, target in zip(outputs,targets):

        losses.append(np.abs(output-target))

    return torch.stack(losses, dim=0).mean()

'''
    Utility Functions
'''

# Utility function to create dataloaders
def make_loaders(batch_size):
    this_dataset = data_factory.dataset_hennepin('train','/u/eag-d1/data/Hennepin/ver10/',
    '/u/eag-d1/data/Hennepin/ver10/hennepin_bbox.csv',
    '/u/eag-d1/data/Hennepin/ver10/hennepin.shp')

    torch.manual_seed(0)

    #print(len(this_dataset))

    train_size = int( np.floor( len(this_dataset) * (1.0-cfg.train.validation_split-cfg.train.test_split) ) )
    val_size = int( np.floor( len(this_dataset) * cfg.train.validation_split ))
    test_size = int(np.floor( len(this_dataset) * cfg.train.test_split ))

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(this_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=cfg.train.shuffle, collate_fn = my_collate,
                             num_workers=cfg.train.num_workers)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn = my_collate,
                             num_workers=cfg.train.num_workers)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn = my_collate,
                             num_workers=cfg.train.num_workers) # need to set the test-loader to test mode here.

    # set the random seed back 
    torch.random.seed()

    return train_loader, val_loader, test_loader

# Minibatch creation for variable size targets in Hennepin Dataset
def my_collate(batch):

    #Masks and values are in lists        
    mask = [item['parcel_masks'] for item in batch]
    value = [item['parcel_values'] for item in batch]

    image = [item['image'].unsqueeze(0) for item in batch]
    image = torch.cat(image)

    return image, mask, value

#Uses torch_grid to make gridded images of batches, not great scaling
def get_grids(model, data_loader):
    with torch.no_grad():
        for batch in data_loader:
            x, mask, value = batch
    
            values = model.get_valOut(x)
            seg = model.cnnOutput(x)

            image_grid = t_util.make_grid(x)
            value_grid = t_util.make_grid(values)
            seg_grid = t_util.make_grid(seg)

            return image_grid, seg_grid, value_grid

   

