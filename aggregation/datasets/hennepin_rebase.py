import os
import pickle
import random

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_function
from PIL import Image
from torch.utils.data import Dataset

from config import cfg


class dataset_hennepin_rebase(Dataset):
    def __init__(self, mode, sample_mode):
        if sample_mode == 'combine' or sample_mode == 'combine_uniform':
            subdir = cfg.data.hennepin.combined_subdir
        else:
            subdir = cfg.data.hennepin.uncombined_subdir
        self.data_dir = os.path.join(cfg.data.hennepin.root_dir, subdir)

        self.mode = mode
        self.sample_mode = sample_mode

        self.val_pkl = os.path.join(self.data_dir, 'vals.pkl')
        self.img_dir = os.path.join(self.data_dir, 'imgs')
        self.mask_dir = os.path.join(self.data_dir, 'masks')

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4712904, 0.36086863, 0.27999857],
                std=[0.24120754, 0.2294313, 0.21295355],
            ),
        ])

        if not os.path.exists(self.val_pkl):
            raise FileNotFoundError(
                f"Hennepin vals.pkl not found at {self.val_pkl}. "
                f"Run scripts/setup_hennepin.sh and/or set HENNEPIN_DATA_ROOT."
            )
        with open(self.val_pkl, 'rb') as f:
            self.vals = pickle.load(f)
        self.vals = np.array(self.vals)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(idx) + ".png")
        mask_path = os.path.join(self.mask_dir, str(idx) + ".pkl")

        img = Image.open(img_path)

        with open(mask_path, 'rb') as f:
            masks = pickle.load(f)
        masks = np.array(masks)

        vals = self.vals[idx]
        vals = torch.tensor(vals)
        masks = torch.tensor(masks)

        if self.mode == 'train':
            img = self.transform(img)

            rotate_param = np.random.randint(1, 5)
            img = transforms_function.rotate(img, 90 * rotate_param, transforms.InterpolationMode.BILINEAR)
            masks = transforms_function.rotate(masks, 90 * rotate_param, transforms.InterpolationMode.BILINEAR)

            if random.random() > 0.5:
                img = transforms_function.hflip(img)
                masks = transforms_function.hflip(masks)

            if random.random() > 0.5:
                img = transforms_function.vflip(img)
                masks = transforms_function.vflip(masks)
        else:
            img = self.transform(img)

        parcel_values = self.vals[idx]

        if self.sample_mode == 'uniform' or self.sample_mode == 'combine_uniform':
            uniform_value_map = np.zeros_like(masks[0])
            total_parcel_mask = np.zeros_like(masks[0])
            for i, mask in enumerate(masks):
                mask = np.array(mask)
                pixel_count = (mask == 1).sum()
                if pixel_count == 0:
                    continue
                uniform_value = parcel_values[i] / pixel_count
                uniform_value_map = np.add(mask * uniform_value, uniform_value_map)

            uniform_value_map = torch.tensor(uniform_value_map).float()
            total_parcel_mask = (uniform_value_map > 0)

            sample = {
                'image': img,
                'total_parcel_mask': total_parcel_mask,
                'uniform_value_map': uniform_value_map,
            }
        else:
            sample = {'image': img, 'masks': masks, 'values': parcel_values}

        return sample

    def __len__(self):
        return len(self.vals)
