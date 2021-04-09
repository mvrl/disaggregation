import torch
import torch.nn as nn
import numpy as np

'''
    RAL Layer from paper
'''
class regionAgg_layer(nn.Module):

    def __init__(self, input_size):
        super(regionAgg_layer, self).__init__()

        # max_regions =  number of regions in the space
        self.input_size = input_size

    def forward(self, x, parcel_masks):
        x = torch.matmul(parcel_masks, x)
        return x;

'''
    Just MSE honestly
'''
def regionAgg_loss(output, target):

    loss = torch.mean((output - target)**2)

    return loss

# https://github.com/orbitalinsight/region-aggregation-public/blob/master/run_cifar10.py
