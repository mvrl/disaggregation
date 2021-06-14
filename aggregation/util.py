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

    def forward(self, x, parcel_mask_batch):
        arr = []

        for i, item in enumerate(parcel_mask_batch):
            #print(torch.tensor(item).dtype)
            #print(x[i].T.dtype)

            arr.append(torch.matmul(x[i], torch.from_numpy(item).T.float().to('cuda')))

        return arr

'''
Loss from the paper?
'''
def MSE(outputs, targets):
    losses = []

    for output,target in zip(outputs,targets):
        losses.append( torch.sum((output - target)**2) ) 

    return torch.stack(losses, dim=0).mean()

# https://github.com/orbitalinsight/region-aggregation-public/blob/master/run_cifar10.py


def MAE(outputs, targets):
    losses = []

    for output, target in zip(outputs,targets):

        losses.append((output-target))

    return torch.stack(losses, dim=0).mean()

def make_loaders(batch_size):
    this_dataset = data_factory.dataset_hennepin('train','/u/eag-d1/data/Hennepin/ver8/',
    '/u/eag-d1/data/Hennepin/ver8/hennepin_bbox.csv',
    '/u/pop-d1/grad/cgar222/Projects/disaggregation/dataset/hennepin_county_parcels/hennepin_county_parcels.shp')

    torch.manual_seed(0)

    print(len(this_dataset))

    train_size = int( np.ceil( len(this_dataset) * (1.0-cfg.train.validation_split-cfg.train.test_split) ) )
    val_size = int( np.floor( len(this_dataset) * cfg.train.validation_split ))
    test_size = int(np.floor( len(this_dataset) * cfg.train.test_split ))

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(this_dataset, [train_size, val_size, test_size])

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
    mask = [item[1] for item in batch]
    value = [item[2] for item in batch]

    image = [item[0].unsqueeze(0) for item in batch]
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

   

