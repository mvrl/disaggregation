from random import sample
import torch
import torchvision.utils as t_util
import torch.nn as nn
import numpy as np
from torch.utils.data.dataloader import DataLoader
from config import cfg
from datasets import hennepin
from datasets import cifar_dataset
import torch.distributions as dist
# CIFAR code from original paperß
# https://github.com/orbitalinsight/region-aggregation-public/blob/master/run_cifar10.py

# This utility handles a lot of the special batching

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

#def poisson_loss(outputs, targets):
#   losses = []
#    for output,target in zip(outputs,targets):
#        poiss = dist.Poisson(output)
#        losses.append( poiss.) 
#    return loss

class gaussAgg_layer(nn.Module):

    def __init__(self):
        super(gaussAgg_layer, self).__init__()

    def forward(self, mean, var, parcel_mask_batch):
        #x: (b, h*w)
        #parcel_mask_batch: (b, num_parc, h*w)
        means = []
        vars = []
        for i, item in enumerate(parcel_mask_batch):
            #item: (num_parc, h*w)
            means.append(torch.matmul(mean[i].cuda(), torch.from_numpy(item).T.float().cuda()))
            vars.append(torch.matmul(var[i].cuda(), torch.from_numpy(item).T.float().cuda()))
        return means, vars

def gaussLoss(means, vars, targets):
    losses = []
    #print(len(means[0]))
    #print(len(vars[0]))
    #print(len(targets[0]))
    for mean,var,target in zip(means,vars,targets):
        std = torch.sqrt(var)
        gauss = dist.Normal(mean, std)
        #print(gauss)
        #print(gauss.log_prob(target))
        losses.append(-torch.sum(gauss.log_prob(target)))
    loss = torch.stack(losses, dim=0).mean()
    return loss

def MSE(outputs, targets):
    losses = []
    for output,target in zip(outputs,targets):
        losses.append( torch.sum((output - target)**2) ) 
    loss = torch.stack(losses, dim=0).mean()
    return loss

def MAE(outputs, targets):
    losses = []
    for output, target in zip(outputs,targets):
        error = output-target
        losses.append(torch.sum(torch.abs(error.cpu())))

    return torch.stack(losses, dim=0).mean()

def make_dataset(mode, sample_mode = ''):
    if cfg.data.name == 'hennepin':
        this_dataset = hennepin.dataset_hennepin(mode, cfg.data.root_dir, sample_mode)
    elif cfg.data.name == 'cifar':
        this_dataset = cifar_dataset.dataset_cifar(mode)
    return this_dataset

def make_loaders( batch_size = cfg.train.batch_size, mode = cfg.mode, sample_mode =cfg.data.hennepin.sample_mode):
    this_dataset = make_dataset(mode, sample_mode)

    torch.manual_seed(0)
    
    train_size = int( np.floor(len(this_dataset) * (1.0-cfg.train.validation_split-cfg.train.test_split) ) )
    val_size = int( np.round( len(this_dataset) * cfg.train.validation_split ))
    test_size = int(np.round( len(this_dataset) * cfg.train.test_split ))


    print(len(this_dataset), len(this_dataset)*0.6, len(this_dataset)*0.2, test_size)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(this_dataset, [train_size, val_size, test_size])


    if(sample_mode == 'uniform' or sample_mode =='uniform_agg'):
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
    mask = [item['masks'] for item in batch]
    value = [item['values'] for item in batch]

    image = [item['image'].unsqueeze(0) for item in batch]
    image = torch.cat(image)

    return image, mask, value

