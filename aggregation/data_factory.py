from albumentations.augmentations.transforms import HorizontalFlip, VerticalFlip
import torch
import numpy as np
import pandas as pd
import random
import os
import PIL
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_function
from torch.utils.data import Dataset
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely
import time
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from geofeather import to_geofeather, from_geofeather

feather_path = '/localdisk0/SCRATCH/watch/hennepin_feathers/'

class dataset_hennepin(Dataset):        # derived from 'dataset_SkyFinder_multi_clean', applies random crop
    def __init__(self, mode, data_dir, csv_path, shp_path, cohens=True):
        
        self.mode = mode
        self.data_dir = data_dir

        self.df = pd.read_csv(csv_path)

        print("Reading GeoDataFrame...")
        t0 = time.time()
        feather_file = os.path.join(feather_path, os.path.basename(shp_path).replace('.shp', '.feather'))
        if os.path.exists(feather_file):
            self.gdf = from_geofeather(feather_file)
        else:
            self.gdf = gpd.read_file(shp_path)
            to_geofeather(self.gdf, feather_file)
        self.shp_path = shp_path
        
        t1 = time.time()
        print(t1-t0, 'secs')
        print((t1-t0)/60, 'mins')

        self.gdf['AVERAGE_MV1'] = self.gdf['TOTAL_MV1'] / self.gdf['geometry'].area
        self.gdf = self.gdf[self.gdf['AVERAGE_MV1'].between(self.gdf['AVERAGE_MV1'].quantile(0.1), self.gdf['AVERAGE_MV1'].quantile(0.9))]
        #Normalize data
        self.gdf['AVERAGE_MV1'] = (self.gdf['AVERAGE_MV1'] - min( self.gdf['AVERAGE_MV1'] )) / ( max(self.gdf['AVERAGE_MV1']) - min(self.gdf['AVERAGE_MV1']))
        print("Done")

        print("Generating list of useful chips")
        self.rows = []
        for index,row in tqdm(self.df.iterrows(), total =len(self.df)):
            dir_path = os.path.join(self.data_dir, str(int(row['lat_mid'])), str(int(row['lon_mid'])))
            masks_dir = os.path.join(dir_path, 'masks')
            if(os.path.isdir(masks_dir)):
                if os.listdir(masks_dir) != []:
                    self.rows.append(row)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet

        self.transforms = A.Compose([
            #A.RandomCrop(height=256, width=256),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            ToTensorV2()
        ])

        self.val_transforms = A.Compose([
            A.RandomCrop(height=256, width=256),
            ToTensorV2()
        ])

        self.cohens = cohens

        self.max_num_parcs = 149
        
    def __len__(self):
        return len(self.rows)

    def getgdf(self):
        return self.gdf

    def __getitem__(self, idx):
        if self.cohens:
            # Grab path from CSV dataframe
            row = self.rows[idx]
            dir_path = os.path.join(self.data_dir, str(int(row['lat_mid'])), str(int(row['lon_mid'])))

            # Load in masks, build aggregation matrix
            masks_dir = os.path.join(dir_path, 'masks')

            masks = []
            values = []

            if(os.path.isdir(masks_dir)):
                if os.listdir(masks_dir) != []:
                    for filename in os.listdir(masks_dir):
                        if(filename.endswith('.tif')):
                            # full file path
                            img_path = os.path.join(masks_dir, filename)

                            #   Grab the PID filename
                            pid = os.path.splitext(filename)[0]
                            
                            # grab the value from the gdf
                            value = self.gdf.loc[ self.gdf['PID'] == pid ]['AVERAGE_MV1'].values.item()

                            # now we grab each mask
                            mask = Image.open(img_path)

                            # ADDDED
                            mask = transforms_function.vflip(mask)

                            masks.append(mask)
                            values.append(value)
    
            # image
            image_name = os.path.join(dir_path, str(int(row['lat_mid']))+'.0_'+str(int(row['lon_mid']))+'.0.tif')
            image = Image.open(image_name)

            
            #Some code for generating the dasymetric maps
            #Grab the bounding box
            row_bbox = (row['lat_min'], row['lon_min'],row['lat_max'], row['lon_max'])
            img_bbox = (row['lat_min'], row['lat_max'],row['lon_min'], row['lon_max']) 
    
            # We should only do this during testing since it really slows things down?
            if(self.mode == 'test'):
                bbox_polygon = shapely.geometry.box(row['lat_min'], row['lon_min'], row['lat_max'], row['lon_max'])
                df2 = gpd.GeoDataFrame(gpd.GeoSeries(bbox_polygon), columns=['geometry'])
                df2.crs = "EPSG:26915"
                polygons = gpd.overlay(self.gdf, df2, how='intersection')
            else:
                polygons = 0
            


            #Then we need to generate a map of the parcels labeled with value
            #dasy = generate_dasymetric_map(polygons, img_bbox)


            # parcel value map
            parcel_fname = os.path.join(dir_path, 'parcel_value.tif')
            value_map = Image.open(parcel_fname)
            value_map = transforms_function.vflip(value_map)
            parcel_fname = os.path.join(dir_path, 'value.tif')


            if self.mode == 'train':            # random flips during training
                if random.random() > 0.5:
                    image = transforms_function.hflip(image)
                    value_map = transforms_function.hflip(value_map)

                    #Add loop for masks
                    masks = [transforms_function.hflip(mask) for mask in masks]
                    
                if random.random() > 0.5:
                    image = transforms_function.vflip(image)
                    value_map = transforms_function.vflip(value_map)

                    #Add loop for masks
                    masks = [transforms_function.vflip(mask) for mask in masks]

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256, 256))
            
            image = transforms_function.crop(image, i, j, h, w)
            value_map = transforms_function.crop(value_map, i, j, h, w)
            masks = [transforms_function.crop(mask, i, j, h, w) for mask in masks]

            image = self.to_tensor(image)
            # note: no ImageNet normalization applied yet

            #Prepare masks here
            
            #for each mask turn to numpy array, flatten, and vstack
            masks = [torch.from_numpy( np.array(mask).flatten()) for mask in masks]

            parcel_masks = np.vstack(masks)
            parcel_values = np.vstack(values)
            
            value_map = torch.from_numpy(np.array(value_map))

            parcel_values = torch.from_numpy(parcel_values)

            sample = {'image': image,'parcel_masks': parcel_masks,
                    'parcel_values':parcel_values,'polygons': polygons,
                    'img_bbox': img_bbox}
        else:
            # Grab path from CSV dataframe
            row = self.rows[idx]
            dir_path = os.path.join(self.data_dir, str(int(row['lat_mid'])), str(int(row['lon_mid'])))
    
            # image
            image_name = os.path.join(dir_path, str(int(row['lat_mid']))+'.0_'+str(int(row['lon_mid']))+'.0.tif')
            image = np.array(Image.open(image_name))

            # Load in masks, build aggregation matrix
            masks_dir = os.path.join(dir_path, 'masks')

            masks = []
            values = []

            num_parcs = 0
            if(os.path.isdir(masks_dir)):
                if os.listdir(masks_dir) != []:
                    for filename in os.listdir(masks_dir):
                        if(filename.endswith('.tif')):
                            # full file path
                            img_path = os.path.join(masks_dir, filename)

                            #   Grab the PID filename
                            pid = os.path.splitext(filename)[0]
                            
                            # grab the value from the gdf
                            value = self.gdf.loc[ self.gdf['PID'] == pid ]['AVERAGE_MV1'].values.item()

                            # now we grab each mask
                            mask = Image.open(img_path)

                            # ADDDED
                            mask = transforms_function.vflip(mask)
                            mask = np.array(mask)

                            masks.append(mask)
                            values.append(value)

                            num_parcs += 1
            
            masks = np.stack(masks, axis=2)
            if masks.shape[2] < self.max_num_parcs:
                pad = np.zeros((masks.shape[0], masks.shape[1], self.max_num_parcs-masks.shape[2]))
                masks = np.concatenate((masks, pad), axis=2)

            values = np.stack(values)
            if values.shape[0] < self.max_num_parcs:
                pad = np.zeros((self.max_num_parcs-values.shape[0],))
                values = np.concatenate((values, pad), axis=0)

            if self.mode == 'train':
                transformed = self.transforms(image=image, mask=masks)
                image = transformed['image']
                masks = transformed['mask']
            else:
                transformed = self.val_transforms(image=image, mask=masks)
                image = transformed['image']
                masks = transformed['mask']

            parcel_masks = masks.view(self.max_num_parcs, -1)
            parcel_values = torch.from_numpy(values)

            sample = {'image': image.float(),'parcel_masks': parcel_masks.float(),
                    'parcel_values':parcel_values, 'num_parcs': num_parcs}
        return sample


def generate_dasymetric_map(polygons, img_bbox):
    fig, ax = plt.subplots()
    polygons.plot(ax=ax, column = 'TOTAL_MV1', alpha = 1, linewidth=3)
    dasy = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    return dasy