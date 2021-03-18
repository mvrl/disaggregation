# A function that returns the desired network

import torch
from config import cfg
import os
from easydict import EasyDict as edict

from torch import nn

def count_trainable_parameters(model):  
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_network(net_name, in_channels=3):
    if net_name == 'unet':
        from models.unet import Unet
        net = Unet(in_channels=3, out_channels=cfg.model.out_channels)
    if net_name == 'nested_unet':
        from models.unet import NestedUNet
        net = NestedUNet(in_channels=3, out_channels=cfg.model.out_channels)
    
    else:
        raise valueError('no model with name:',net_name)
    
    # set to CUDA and GPUs
    net.cuda()
    if len(cfg.train.device_ids)>1:
        net = nn.DataParallel(net, device_ids=cfg.train.device_ids)
    
    print('model parameters:', count_trainable_parameters(net))
    
    return net