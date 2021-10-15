import pickle
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

from geofeather import to_geofeather, from_geofeather

class dataset_hennepin(Dataset):        # derived from 'dataset_SkyFinder_multi_clean', applies random crop
    def __init__(self, mode, data_dir):
        
        self.mode = mode
        self.data_dir = data_dir

        #handling paths
        csv_path = os.path.join(self.data_dir, 'hennepin_bbox.csv')
        shp_path = os.path.join(self.data_dir, 'hennepin.shp')
        feather_path = os.path.join(self.data_dir, 'hennepin.feather')

        values_pickle_file = os.path.join(self.data_dir, 'hennepin_vals.pkl')
        masks_pickle_file = os.path.join(self.data_dir, 'hennepin_masks.pkl')
        paths_pickle_file = os.path.join(self.data_dir, 'hennepin_paths.pkl')

        self.df = pd.read_csv(csv_path)

        # Reading the GDF here is a little redundant, 
        print("Reading GeoDataFrame...")
        if os.path.exists(feather_path):
            self.gdf = from_geofeather(feather_path)
        else:
            self.gdf = gpd.read_file(shp_path)
            to_geofeather(self.gdf, feather_path)
        self.shp_path = shp_path

        # this will moreso be used for plotting
        self.gdf['AVERAGE_MV1'] = self.gdf['TOTAL_MV1'] / self.gdf['geometry'].area

        '''
            NORMAlIZATION
        '''
        #self.gdf['TOTAL_MV1']= self.gdf['TOTAL_MV1'].clip(self.gdf['TOTAL_MV1'].quantile(0.05),self.gdf['TOTAL_MV1'].quantile(0.95))
        self.gdf = self.gdf[self.gdf['TOTAL_MV1'].between(self.gdf['TOTAL_MV1'].quantile(0.1), self.gdf['TOTAL_MV1'].quantile(0.9))]
        #Normalize data
        self.gdf['TOTAL_MV1'] = (self.gdf['TOTAL_MV1'] - min( self.gdf['TOTAL_MV1'] )) / ( max(self.gdf['TOTAL_MV1']) - min(self.gdf['TOTAL_MV1'])) * 1000 
        #self.gdf['TOTAL_MV1'] = (self.gdf['TOTAL_MV1'] - self.gdf['TOTAL_MV1'].mean()) / self.gdf['TOTAL_MV1'].std()
        print("Done")

        #Un-used Currently
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet

        print("Loading all values...")
        self.all_values = []
        self.all_mask_paths = []
        self.all_rows = []

        if os.path.exists(values_pickle_file):
            with open(values_pickle_file, 'rb') as f:
                self.all_values = pickle.load(f)
            with open(masks_pickle_file, 'rb') as f:
                self.all_mask_paths = pickle.load(f)
            with open(paths_pickle_file, 'rb') as f:
                self.all_rows = pickle.load(f)
        else:
            for index,row in tqdm(self.df.iterrows(), total =len(self.df)):
                # Grab path from CSV dataframe
                dir_path = os.path.join(self.data_dir, str(int(row['lat_mid'])), str(int(row['lon_mid'])))

                # Load in masks, build aggregation matrix
                masks_dir = os.path.join(dir_path, 'masks')
                values = []
                masks = []

                if(os.path.isdir(masks_dir)):
                    for filename in os.listdir(masks_dir):
                        if(filename.endswith('.tif')):

                            img_path = os.path.join(masks_dir, filename)

                            #   Grab the PID filename
                            pid = os.path.splitext(filename)[0]
                            
                            # get parcel
                            parcel = self.gdf.loc[self.gdf['PID'] == pid]
                            # grab the value from the gdf
                            if(len(parcel['TOTAL_MV1'].values) == 1):
                                value = parcel['TOTAL_MV1'].values.item()
                            else:
                                #print("skipping a thing")
                                continue
                            
                            values.append(value)
                            masks.append(img_path)
                if values != []:
                    values = np.stack(values)
                    self.all_values.append(values)
                    self.all_mask_paths.append(masks)
                    self.all_rows.append(row)

            with open(values_pickle_file, 'wb') as f:
                pickle.dump(self.all_values, f)
            with open(masks_pickle_file, 'wb') as f:
                pickle.dump(self.all_mask_paths, f)
            with open(paths_pickle_file, 'wb') as f:
                pickle.dump(self.all_rows, f)
        
    def __len__(self):
        return len(self.all_rows)

    def getgdf(self):
        return self.gdf
    
    def set_mode(self, mode):
        self.mode = mode

    def __getitem__(self, idx):
        # Grab path from CSV dataframe
        row = self.all_rows[idx]
        dir_path = os.path.join(self.data_dir, str(int(row['lat_mid'])), str(int(row['lon_mid'])))

        masks = []
        for mask_path in self.all_mask_paths[idx]:
            # now we grab each mask
            mask = Image.open(mask_path)

            # each mask is generated upside down
            mask = transforms_function.vflip(mask)

            masks.append(mask)

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

        if self.mode == 'train':            # random flips during training
            if random.random() > 0.5:
                image = transforms_function.hflip(image)
                #value_map = transforms_function.hflip(value_map)

                #Add loop for masks
                masks = [transforms_function.hflip(mask) for mask in masks]
                
            if random.random() > 0.5:
                image = transforms_function.vflip(image)
                #value_map = transforms_function.vflip(value_map)

                #Add loop for masks
                masks = [transforms_function.vflip(mask) for mask in masks]

        # Random crop
        #i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256, 256))
        
        #image = transforms_function.crop(image, i, j, h, w)
        #value_map = transforms_function.crop(value_map, i, j, h, w)
        #masks = [transforms_function.crop(mask, i, j, h, w) for mask in masks]

        image = self.to_tensor(image)
        # note: no ImageNet normalization applied yet

        #Prepare masks here
        
        #for each mask turn to numpy array, flatten, and vstack
        masks = [torch.from_numpy( np.array(mask).flatten()) for mask in masks]

        parcel_masks = np.vstack(masks)
        parcel_values = self.all_values[idx] #np.vstack(values)
        
        #value_map = torch.from_numpy(np.array(value_map))

        parcel_values = torch.from_numpy(parcel_values)

        sample = {'image': image,'parcel_masks': parcel_masks,
                'parcel_values':parcel_values,'polygons': polygons,
                'img_bbox': img_bbox}
        return sample


def generate_dasymetric_map(polygons, img_bbox):
    fig, ax = plt.subplots()
    polygons.plot(ax=ax, column = 'TOTAL_MV1', alpha = 1, linewidth=3)
    dasy = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    return dasy