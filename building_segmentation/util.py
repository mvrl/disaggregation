import torch
import torch.nn as nn
import numpy as np

'''
    This layer will have a lot of weights

    437245 Regions

    CNN output = 224*224 = 50,176 pixels
'''
class regionAgg_layer(nn.Module):

    def __init__(self, input_size, max_regions):
        super(regional_aggregation_layer, self).__init__()

        # max_regions =  number of regions in the space
        self.max_regions = max_regions;
        self.input_size = input_size

        self.layer = nn.Linear(input_size,  max_regions)

    def forward(self, x):
        x = layer(x)
        return x;

'''
    We need a loss that will ignore the regions not contained in the
    input image.

    contained_regions are indices of the output array
'''
def regionAgg_loss(output, target, contained_regions):

    # np.take indexes the output
    output = np.take(output,contained_regions)

    loss = torch.mean((output - target)**2)

    return loss
