import torch
import torch.nn as nn
import numpy as np

'''
    This layer will have a lot of weights

    437245 Regions

    CNN output = 224*224 = 50,176 pixels
'''
class regionAgg_layer(nn.Module):

    def __init__(self, input_size, agg_matrix):
        super(regionAgg_layer, self).__init__()

        # max_regions =  number of regions in the space
        self.agg_matrix = agg_matrix;
        self.input_size = input_size

    def forward(self, x):
        x = torch.matmul(agg_matrix['mask'], x)
        return x;

'''
    We need a loss that will ignore the regions not contained in the
    input image.

    contained_regions are are geometries/values from the loaded area
'''
def regionAgg_loss(output, agg_matrix, parcel_ids):

    #find the indexes of the contained regions in the aggregation matrix
    target = numpy.array( agg_matrix['value'] ) 

    indices = []

    for id in parcel_ids:
        indices.append( df.index[df['PID'] == id])

    # np.take indexes the output and target
    output = np.take(output,indices)
    target = np.take(target,indices)

    loss = torch.mean((output - target)**2)

    return loss
