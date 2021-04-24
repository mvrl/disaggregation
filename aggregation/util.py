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

    def forward(self, x, parcel_mask_batch):
        arr = []

        for i, item in enumerate(parcel_mask_batch):
            #print(torch.tensor(item).dtype)
            #print(x[i].T.dtype)

            arr.append(torch.matmul(x[i], torch.tensor(item).T.float()))

        return arr

'''
    This takes lists of tensors, where each tensor is variable size. 
    The length of each list is the batch size
'''
def regionAgg_loss(outputs, targets):
    losses = []

    for output,target in zip(outputs,targets):
        #print(output)
        #print(target)

        #target = torch.tensor(target)

        losses.append( torch.mean((output - target)**2) ) 

    return torch.stack(losses, dim=0).mean()

# https://github.com/orbitalinsight/region-aggregation-public/blob/master/run_cifar10.py
