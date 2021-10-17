import numpy as np
import torch
import util
import torchvision.transforms as transforms
from tqdm import tqdm

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    train_loader, val_loader, test_loader = util.make_loaders(batch_size = 1, mode = 'train')

    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    for i, data in tqdm(enumerate(train_loader, 0)):
        # shape (batch_size, 3, height, width)

        image, mask, value = data
        numpy_image = image.numpy()
        
        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std0 = np.std(numpy_image, axis=(0,2,3))
        batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
        
        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        pop_std1.append(batch_std1)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    pop_std1 = np.array(pop_std1).mean(axis=0)

    print(pop_mean, pop_std0, pop_std1)