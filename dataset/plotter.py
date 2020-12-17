import matplotlib.pyplot as plt
import numpy as np

'''
Plotter for channel first images
'''
def plot_tile(ax, image_array,extent):

    image = np.moveaxis(image_array, 0, -1)

    ax.imshow(image, extent = extent, origin = 'upper')