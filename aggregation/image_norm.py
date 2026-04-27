import numpy as np
import torch
import util
import torchvision.transforms as transforms
from tqdm import tqdm

if __name__ == "__main__":
    #These transforms must be applied somewhere to count correctly.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    train_loader, val_loader, test_loader = util.make_loaders(batch_size = 1, mode = 'train')

    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    pixel_mean =[]
    pixel_std = []
    for i, data in tqdm(enumerate(train_loader, 0)):
        # shape (batch_size, 3, height, width)

        image, masks, values = data['image'], data['masks'], data['values']
        numpy_image = image.numpy()
        
        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std0 = np.std(numpy_image, axis=(0,2,3))
        batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)

        pixel_mean_i = np.mean(numpy_image)
        pixel_std_i = np.std(numpy_image)
        
        pixel_mean.append(pixel_mean_i)
        pixel_std.append(pixel_std_i)

        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        pop_std1.append(batch_std1)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    pop_std1 = np.array(pop_std1).mean(axis=0)

    min_mean = np.array(pop_mean).min(axis=0)
    min_std0 = np.array(pop_std0).min(axis=0)
    min_std1 = np.array(pop_std1).min(axis=0)

    max_mean = np.array(pop_mean).max(axis=0)
    max_std0 = np.array(pop_std0).max(axis=0)
    max_std1 = np.array(pop_std1).max(axis=0)

    pixel_mean = np.array(pixel_mean).mean()
    min_pixel_std = np.array(pixel_std).min()
    max_pixel_std = np.array(pixel_std).max()

    print(pixel_mean,min_pixel_std, max_pixel_std)

    print(min_mean, max_mean, min_std0, max_std0)

    print(pop_mean, pop_std0, pop_std1)