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
        #x: (b, h*w)
        #parcel_mask_batch: (b, num_parc, h*w)
        arr = []
        for i, item in enumerate(parcel_mask_batch):
            #item: (num_parc, h*w)
            arr.append(torch.matmul(x[i].cuda(), torch.from_numpy(item).T.float().cuda()))
        return arr



#class chip_value_sum(nn.Module):

#    def __init__(self):
#       super(chip_value_sum, self).__init__()

#    def forward(self, x, parcel_mask_batch):
#        arr = []
#        for i, item in enumerate(parcel_mask_batch):
#            #item: (num_parc, h*w)
#            arr.append(torch.matmul(x[i], torch.from_numpy(item).T.float().to('cuda')))


'''
Loss from the paper?
'''
def MSE(outputs, targets):
    losses = []
    for output,target in zip(outputs,targets):
        losses.append( torch.sum((output - target)**2) ) 
    loss = torch.stack(losses, dim=0).mean()
    return loss

# https://github.com/orbitalinsight/region-aggregation-public/blob/master/run_cifar10.py


def MAE(outputs, targets):
    losses = []

    for output, target in zip(outputs,targets):
        error = output-target
        losses.append(torch.sum(torch.abs(error.cpu())))

    return torch.stack(losses, dim=0).mean()

def make_dataset(mode, uniform = False):
    this_dataset = data_factory.dataset_hennepin(mode, cfg.data.root_dir, uniform)
    return this_dataset

def make_loaders( batch_size = cfg.train.batch_size, mode = cfg.mode, uniform = cfg.uniform):
    this_dataset = make_dataset(mode, uniform)

    torch.manual_seed(0)

    train_size = int( np.round(len(this_dataset) * (1.0-cfg.train.validation_split-cfg.train.test_split) ) )
    val_size = int( np.round( len(this_dataset) * cfg.train.validation_split ))
    test_size = int(np.round( len(this_dataset) * cfg.train.test_split ))


    print(len(this_dataset), len(this_dataset)*0.6, len(this_dataset)*0.2, test_size)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(this_dataset, [train_size, val_size, test_size])


    if(uniform):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=cfg.train.shuffle,
                            num_workers=cfg.train.num_workers)

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=cfg.train.num_workers)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=cfg.train.num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=cfg.train.shuffle, collate_fn = my_collate,
                            num_workers=cfg.train.num_workers)

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn = my_collate,
                            num_workers=cfg.train.num_workers)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn = my_collate,
                            num_workers=cfg.train.num_workers)

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

# Minibatch creation for variable size targets in Hennepin Dataset
def test_collate(batch):

    #Masks and values are in lists        
    mask = [item['parcel_masks'] for item in batch]
    value = [item['parcel_values'] for item in batch]

    image = [item['image'].unsqueeze(0) for item in batch]
    image = torch.cat(image)

    polygons = [item['polygons'] for item in batch]
    img_bbox = [item['img_bbox'] for item in batch]

    return image, mask, value, polygons, img_bbox


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

   

