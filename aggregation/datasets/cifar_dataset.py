import torchvision
import numpy as np
import cv2
import torch
from scipy.spatial.distance import cdist
from torchvision import transforms
from sklearn.utils import check_random_state
from torch.utils.data import Dataset

# Adapted from
# https://github.com/orbitalinsight/region-aggregation-public/blob/master/run_cifar10.py

def create_random_voronoi(size, num_regions):
    """
    Creates a random voronoi image of a certain width and height
    by sampling num_region random points. Returns int64 singe channel image.
    Code adapted from:
            https://www.learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/
    """
    print(size)
    w = h = size  # For this project, assume images are square
    rect = (0, 0, w, h)
    seed = 0
    rng = check_random_state(seed)
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

def prep_y_simple(X):
    """ Creates a simple target Y tensor, in this case, binary."""
    N,d,h,w = X.shape
    num_centroids = 15
    threshold = 0.2
    seed = 0
    rng = check_random_state(seed)
    color_vecs = X.permute(0,2,3,1).reshape(-1,3)
    source_colors = color_vecs[rng.choice(color_vecs.shape[0],num_centroids),:]
    tmp = cdist(source_colors, color_vecs)
    tmp = np.min(tmp, axis=0) < threshold
    Y = (tmp).reshape(N,1,h,w).astype(np.float32)
    return Y


class dataset_cifar(Dataset):
    def __init__(self, mode):
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.dataset = []

        if(mode == 'train'):
            self.dataset = torchvision.datasets.CIFAR10(root = './', train=True, download=True, transform=None)
        else:
            self.dataset = torchvision.datasets.CIFAR10(root = './', train=False, download=True, transform=None)

        self.max_regions = 10

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        voronoi = create_random_voronoi(32, self.max_regions)
        masks = torch.nn.functional.one_hot(torch.tensor(voronoi),10)
        image_tensor = self.transform(sample[0])
        Y = prep_y_simple(torch.tensor(image_tensor.unsqueeze(0)))
        masks = masks.permute(-1, 0 ,1).flatten(1)

        #region aggregation
        region_sums = torch.matmul(Y.flatten(), masks.T.float()) 

        sample = {'image': image_tensor,'masks': masks,
                'values':region_sums}


