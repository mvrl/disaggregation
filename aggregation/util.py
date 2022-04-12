from random import sample
import torch
import torchvision.utils as t_util
import torch.nn as nn
import numpy as np
from torch.utils.data.dataloader import DataLoader
from config import cfg
from datasets import hennepin_rebase
import torch.distributions as dist


def make_dataset(mode, sample_mode = cfg.data.hennepin.sample_mode):
    if cfg.data.name == 'hennepin':
        this_dataset = hennepin_rebase.dataset_hennepin_rebase(mode, sample_mode)
    return this_dataset

def make_loaders( batch_size = cfg.train.batch_size, mode = cfg.mode, sample_mode =cfg.data.hennepin.sample_mode):
    this_dataset = make_dataset(mode, sample_mode)

    torch.manual_seed(0)
    
    train_size = int( np.floor(len(this_dataset) * (1.0-cfg.train.validation_split-cfg.train.test_split) ) )
    val_size = int( np.floor( len(this_dataset) * cfg.train.validation_split ))
    test_size = int(np.ceil( len(this_dataset) * cfg.train.test_split ))


    print(len(this_dataset), len(this_dataset)*0.8, len(this_dataset)*0.1, test_size)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(this_dataset, [train_size, val_size, test_size])

    #make sure we use the full set for validation on combination experiments
    #if sample_mode == 'combine':
    #    val_dataset.sample_mode = ''
    #if sample_mode == 'combine_uniform':
    #    val_dataset.sample_mode = 'uniform'

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=cfg.train.shuffle,
                            num_workers=cfg.train.num_workers)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=cfg.train.num_workers)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=cfg.train.num_workers)

    return train_loader, val_loader, test_loader
