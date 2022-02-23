import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from scipy.spatial.distance import cdist
from torchvision import transforms
from sklearn.utils import check_random_state
from torch.utils.data import Dataset, DataLoader

def create_random_voronoi(size, num_regions, rng):
    """
    Creates a random voronoi image of a certain width and height
    by sampling num_region random points. Returns int64 singe channel image.
    Code adapted from:
            https://www.learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/
    """
    w = h = size  # For this project, assume images are square
    rect = (0, 0, w, h)
    
    subdiv = cv2.Subdiv2D(rect)
    for rp in rng.randint(0, w, size=(num_regions, 2)):
        subdiv.insert((rp[0], rp[1]))
    facets, centers = subdiv.getVoronoiFacetList([])
    img = np.zeros((w, h), dtype=np.uint8)
    c = 0
    for i in range(0,len(facets)):
        ifacet_arr = []
        for f in facets[i]:
            ifacet_arr.append(f)
        ifacet = np.array(ifacet_arr, np.int)
        cv2.fillConvexPoly(img, ifacet, c, cv2.INTER_NEAREST, 0)  # Do not use another interpolation!!
        c += 1
    return img.astype(np.int64)

def prep_y_simple(X, centroids):
    """ Creates a simple target Y tensor, in this case, binary."""
    N,d,h,w = X.shape
    num_centroids = 15
    threshold = 0.2
    seed = 0
    rng = check_random_state(seed)
    color_vecs = X.permute(0,2,3,1).reshape(-1,3)
    source_colors = color_vecs[centroids,:]
    tmp = cdist(source_colors, color_vecs)
    tmp = np.min(tmp, axis=0) < threshold
    Y = (tmp).reshape(N,1,h,w).astype(np.float32)
    return Y

def get_centroids():
    """ Creates a simple target Y tensor, in this case, binary."""
    shape = 1024 # HARDCODED
    num_centroids = 15
    seed = 0
    rng = check_random_state(seed)
    centroids = rng.choice(shape,num_centroids)
    return centroids

class dataset_cifar(Dataset):
    def __init__(self, mode):
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.dataset = []

        self.mode = mode

        if(self.mode == 'train'):
            self.dataset = torchvision.datasets.CIFAR10(root = './', train=True, download=True, transform=None)
        else:
            self.dataset = torchvision.datasets.CIFAR10(root = './', train=False, download=True, transform=None)
            
        self.centroids = get_centroids()

        self.max_regions = 10

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        seed = idx
        rng = check_random_state(seed)
        voronoi = create_random_voronoi(32, self.max_regions, rng)

        masks = torch.nn.functional.one_hot(torch.tensor(voronoi),10)

        image_tensor = self.transform(sample[0])
        Y = prep_y_simple(image_tensor.unsqueeze(0), self.centroids)
        masks = masks.permute(-1, 0 ,1).flatten(1)
        Y = torch.tensor(Y).squeeze(0)

        #region aggregation
        region_sums = torch.matmul(Y.flatten(), masks.T.float())

        uniform_map = np.zeros_like(masks[0])
        total_parcel_mask = np.zeros_like(masks[0])
        for i,mask in enumerate(masks):
            mask = np.array(mask)
            pixel_count = (mask == 1).sum()
            #Some are zero? this is creating nans
            if(pixel_count == 0):
                pixel_count = 1
            uniform_value = np.array( region_sums[i]/pixel_count )
            #print(uniform_value)
            uniform_map = np.add(mask*uniform_value, uniform_map)

        uniform_map = torch.reshape(torch.tensor(uniform_map), (32,32))

        sample = {'image': image_tensor,'voronoi': voronoi, 'masks': masks,
                'values':region_sums, 'uniform': uniform_map, 'Y': Y}
        return sample
