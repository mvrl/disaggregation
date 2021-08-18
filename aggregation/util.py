import torch
import torchvision.utils as t_util
import torch.nn as nn
import numpy as np
from torch.utils.data.dataloader import DataLoader
from config import cfg
import data_factory
import matplotlib.pyplot as plt
''' 
    RAL Layer from paper
'''
class regionAgg_layer(nn.Module):

    def __init__(self):
        super(regionAgg_layer, self).__init__()

    def forward(self, x, parcel_mask_batch, cohens=True, num_parcs=None):
        #x: (b, h*w)
        #parcel_mask_batch: (b, num_parc, h*w)
        if cohens:
            arr = []
            for i, item in enumerate(parcel_mask_batch):
                #item: (num_parc, h*w)
                arr.append(torch.matmul(x[i], torch.from_numpy(item).T.float().to('cuda')))
        else:
            b = x.shape[0]
            hw = x.shape[2]
            block_masks = torch.block_diag(*[ parcel_mask_batch[i, :num_parcs[i], :] for i in range(b) ]).reshape(b, hw, -1)
            arr = torch.bmm(x, block_masks)
        return arr

'''
Loss from the paper?
'''
def MSE(outputs, targets, cohens=True, num_parcs=None):
    if cohens:
        losses = []
        for output,target in zip(outputs,targets):
            losses.append( torch.sum((output - target)**2) ) 
        loss = torch.stack(losses, dim=0).mean()
    else:
        b = outputs.shape[0]
        block_parcel_values = torch.block_diag(*[ targets[i, :num_parcs[i]] for i in range(b) ])
        loss = torch.sum((outputs.squeeze() - block_parcel_values)**2, dim=1).mean()
    return loss

# https://github.com/orbitalinsight/region-aggregation-public/blob/master/run_cifar10.py


def MAE(outputs, targets):
    losses = []

    for output, target in zip(outputs,targets):

        losses.append(np.abs(output-target))

    return torch.stack(losses, dim=0).mean()

def make_loaders(cohens):
    this_dataset = data_factory.dataset_hennepin('train','/u/eag-d1/data/Hennepin/ver8/',
    '/u/eag-d1/data/Hennepin/ver8/hennepin_bbox.csv',
    '/u/eag-d1/data/Hennepin/hennepin_county_parcels/hennepin_county_parcels.shp', cohens)

    torch.manual_seed(0)

    train_size = int( np.ceil( len(this_dataset) * (1.0-cfg.train.validation_split-cfg.train.test_split) ) )
    val_size = int( np.floor( len(this_dataset) * cfg.train.validation_split ))
    test_size = int(np.floor( len(this_dataset) * cfg.train.test_split ))

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(this_dataset, [train_size, val_size, test_size])

    if cohens:
        train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=cfg.train.shuffle, collate_fn = my_collate,
                                num_workers=cfg.train.num_workers)

        val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, collate_fn = my_collate,
                                num_workers=cfg.train.num_workers)

        test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False, collate_fn = my_collate,
                                num_workers=cfg.train.num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=cfg.train.shuffle,
                                num_workers=cfg.train.num_workers)

        val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False,
                                num_workers=cfg.train.num_workers)

        test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False,
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

   

