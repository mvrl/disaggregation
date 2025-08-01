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
    def __init__(self, mode, data_dir, sample_mode):
        
        self.mode = mode
        self.data_dir = data_dir
        self.sample_mode = sample_mode
        random.seed(0)

        #handling paths
        csv_path = os.path.join(self.data_dir, 'hennepin_bbox.csv')
        shp_path = os.path.join(self.data_dir, 'hennepin.shp')
        feather_path = os.path.join(self.data_dir, 'hennepin.feather')

        values_pickle_file = os.path.join(self.data_dir, 'hennepin_vals.pkl')
        masks_pickle_file = os.path.join(self.data_dir, 'hennepin_masks.pkl')
        paths_pickle_file = os.path.join(self.data_dir, 'hennepin_paths.pkl')
        pids_pickle_file = os.path.join(self.data_dir, 'hennepin_pids.pkl')

        self.df = pd.read_csv(csv_path)

        # Reading the GDF here is a little redundant, 
        print("Reading GeoDataFrame...")
        if os.path.exists(feather_path):
            self.gdf = from_geofeather(feather_path)
        else:
            self.gdf = gpd.read_file(shp_path)
            to_geofeather(self.gdf, feather_path)
        self.shp_path = shp_path
        print("Done")

        # We are going to use this instead of the value for getting outliers
        self.gdf['AVERAGE_MV1'] = self.gdf['TOTAL_MV1'] / self.gdf['geometry'].area

        '''
            Label Normalization
        '''
        # Watch for outliers
        self.gdf = self.gdf[self.gdf['TOTAL_MV1'].between(self.gdf['TOTAL_MV1'].quantile(0.1), self.gdf['TOTAL_MV1'].quantile(0.9))]
        self.gdf = self.gdf[self.gdf['AVERAGE_MV1'].between(self.gdf['AVERAGE_MV1'].quantile(0.1), self.gdf['AVERAGE_MV1'].quantile(0.9))]
        
        # Old Min-Max
        #self.gdf['TOTAL_MV1'] = (self.gdf['TOTAL_MV1'] - min( self.gdf['TOTAL_MV1'] )) / ( max(self.gdf['TOTAL_MV1']) - min(self.gdf['TOTAL_MV1'])) * 1000
        
        # old clipping code
        #self.gdf['TOTAL_MV1']= self.gdf['TOTAL_MV1'].clip(self.gdf['TOTAL_MV1'].quantile(0.05),self.gdf['TOTAL_MV1'].quantile(0.95))

        #standardization
        #self.gdf['TOTAL_MV1'] = (self.gdf['TOTAL_MV1'] - self.gdf['TOTAL_MV1'].mean()) / self.gdf['TOTAL_MV1'].std()

        #Image Transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4712904,0.36086863,0.27999857], std=[0.24120754, 0.2294313, 0.21295355])
        ])

        self.to_tensor = transforms.ToTensor()
        
        print("Loading all values...")
        self.all_values = []
        self.all_mask_paths = []
        self.all_rows = []
        self.all_pids = []

        if os.path.exists(values_pickle_file):
            with open(values_pickle_file, 'rb') as f:
                self.all_values = pickle.load(f)
            with open(masks_pickle_file, 'rb') as f:
                self.all_mask_paths = pickle.load(f)
            with open(paths_pickle_file, 'rb') as f:
                self.all_rows = pickle.load(f)
            with open(pids_pickle_file, 'rb') as f:
                self.all_pids = pickle.load(f)
        else:
            for index,row in tqdm(self.df.iterrows(), total =len(self.df)):
                # Grab path from CSV dataframe
                dir_path = os.path.join(self.data_dir, str(int(row['lat_mid'])), str(int(row['lon_mid'])))

                # Load in masks, build aggregation matrix
                masks_dir = os.path.join(dir_path, 'masks')
                values = []
                pids = []
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
                            pids.append(pid)
                            masks.append(img_path)

                if values != []:
                    values = np.stack(values)
                    self.all_values.append(values)
                    self.all_mask_paths.append(masks)
                    self.all_rows.append(row)
                    self.all_pids.append(pids)

            with open(values_pickle_file, 'wb') as f:
                pickle.dump(self.all_values, f)
            with open(masks_pickle_file, 'wb') as f:
                pickle.dump(self.all_mask_paths, f)
            with open(paths_pickle_file, 'wb') as f:
                pickle.dump(self.all_rows, f)
            with open(pids_pickle_file, 'wb') as f:
                pickle.dump(self.all_pids, f)

        print("Done...")
        
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

        # *** Maybe it makes sense to do this in initialization ***
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

        # Fetch the Polygons
        # We should only do this during testing/visualization since it really slows things down
        if(self.mode == 'vis'):
            bbox_polygon = shapely.geometry.box(row['lat_min'], row['lon_min'], row['lat_max'], row['lon_max'])
            df2 = gpd.GeoDataFrame(gpd.GeoSeries(bbox_polygon), columns=['geometry'])
            df2.crs = "EPSG:26915"
            polygons = gpd.overlay(self.gdf, df2, how='intersection')
        else:
            polygons = 0

        #Random flips, train datas augmentation
        if self.mode == 'train':          

            if random.random() > 0.5:
                image = transforms_function.hflip(image)
                masks = [transforms_function.hflip(mask) for mask in masks]
                
            if random.random() > 0.5:
                image = transforms_function.vflip(image)
                masks = [transforms_function.vflip(mask) for mask in masks]

            if random.random() > 0.5:
                image = transforms_function.rotate(image, 90, transforms.InterpolationMode.BILINEAR)
                masks = [transforms_function.rotate(mask, 90, transforms.InterpolationMode.BILINEAR) for mask in masks]

            if random.random() > 0.5:
                image = transforms_function.rotate(image, 270, transforms.InterpolationMode.BILINEAR)
                masks = [transforms_function.rotate(mask, 270, transforms.InterpolationMode.BILINEAR) for mask in masks]

        #normalization
        if self.mode != 'vis':
            #Apply Transformation
            image = self.transform(image)
        else:
            image = self.to_tensor(image)

        #Grab the values
        parcel_values = self.all_values[idx] #np.vstack(values)
       
        if(self.sample_mode == 'uniform'):
            uniform_value_map = np.zeros_like(masks[0])
            total_parcel_mask = np.zeros_like(masks[0])
            for i,mask in enumerate(masks):
                mask = np.array(mask)
                pixel_count = (mask == 1).sum()
                uniform_value = parcel_values[i]/pixel_count
                uniform_value_map = np.add(mask*uniform_value, uniform_value_map)

            total_parcel_mask = (uniform_value_map > 0)
            
            sample = {'image':image, 'total_parcel_mask':total_parcel_mask,
                        'uniform_value_map': uniform_value_map}

        #randomly select, and combine parcels with their nearest neighbor
        elif(self.sample_mode == 'combine'):
            #Fetch pids, we will need the gdf index
            pids = self.all_pids[idx]
            masks = [torch.from_numpy( np.array(mask).flatten()) for mask in masks]
            
            parcel_values = torch.from_numpy(parcel_values)

            if(len(pids) > 1):

                geoms = []

                for pid in pids:
                    geoms.append( self.gdf.loc[self.gdf['PID'] == pid]['geometry'].values[0] )
                
                combine_length = int(len(geoms) / 2)
                for x in range(combine_length):
                    #randomly select index
                    randomINDEX = random.randint(0, len(geoms) - 1)
                    parcel_geom = geoms[randomINDEX]
                    #print(parcel_geom)
                    others = geoms.copy()
                    del others[randomINDEX]
                    others = gpd.GeoSeries(others)

                    #print(others.distance(parcel_geom) )
                    distances = others.distance(parcel_geom)
                    min_value = np.min( distances )
                    #print(others.index(min_value))
                    closestGEOM = others[distances == min_value].values[0]
                    #print(closestGEOM)
                    closestINDEX = geoms.index(closestGEOM)

                    #print("RANDOM_INDEX: ", randomINDEX)
                    #print("closest_index: ", closestINDEX)
                    
                    mask_sum = masks[closestINDEX] + masks[randomINDEX]
                    val_sum = parcel_values[closestINDEX] + parcel_values[randomINDEX]

                    if(parcel_values[closestINDEX] / np.count_nonzero(masks[closestINDEX]) < 1000):
                        
                        parcel_values[closestINDEX] = val_sum
                        masks[closestINDEX] = mask_sum

                        del geoms[randomINDEX]
                        del masks[randomINDEX]
                        #masks = np.delete(masks,randomINDEX)
                        parcel_values = np.delete(parcel_values,randomINDEX)
                    
                    
            #Masks will be reduced by n/2
            #values will be reduced by n/2
            # TODO polygon combination, unimplemented
            #bbox stays the same
            masks = np.vstack(masks)

            sample = {'image': image,'masks': masks,
                'values':parcel_values,'polygons': polygons,
                'img_bbox': img_bbox}
        #randomly select, and combine parcels with their nearest neighbor
        elif(self.sample_mode == 'combine_uniform'):
            #Fetch pids, we will need the gdf index
            pids = self.all_pids[idx]
            masks = [torch.from_numpy( np.array(mask)) for mask in masks]
            
            #parcel_values = torch.from_numpy(parcel_values)

            if(len(pids) > 1):

                geoms = []

                for pid in pids:
                    geoms.append( self.gdf.loc[self.gdf['PID'] == pid]['geometry'].values[0] )
                
                combine_length = int(len(geoms) / 2)
                for x in range(combine_length):
                    #randomly select index
                    randomINDEX = random.randint(0, len(geoms) - 1)
                    parcel_geom = geoms[randomINDEX]
                    #print(parcel_geom)
                    others = geoms.copy()
                    del others[randomINDEX]
                    others = gpd.GeoSeries(others)


                    #print(others.distance(parcel_geom) )
                    distances = others.distance(parcel_geom)
                    min_value = np.min( distances )
                    #print(others.index(min_value))
                    closestGEOM = others[distances == min_value].values[0]
                    #print(closestGEOM)
                    closestINDEX = geoms.index(closestGEOM)

                    mask_sum = masks[closestINDEX] + masks[randomINDEX]
                    val_sum = parcel_values[closestINDEX] + parcel_values[randomINDEX]

                    if(parcel_values[closestINDEX] / np.count_nonzero(masks[closestINDEX]) < 1000):
                        
                        parcel_values[closestINDEX] = val_sum
                        masks[closestINDEX] = mask_sum

                        del geoms[randomINDEX]
                        del masks[randomINDEX]
                        #masks = np.delete(masks,randomINDEX)
                        parcel_values = np.delete(parcel_values,randomINDEX)
                    
                    
            #Masks will be reduced by n/2
            #values will be reduced by n/2
            # TODO polygon combination, unimplemented
            #bbox stays the same
            #masks = np.vstack(masks)

            uniform_value_map = np.zeros_like(masks[0])
            total_parcel_mask = np.zeros_like(masks[0])
            pixel_count_sum = 0
            parcel_values_sum = 0
            for i,mask in enumerate(masks):
                mask = np.array(mask)
                pixel_count = (mask == 1).sum()
                pixel_count_sum += pixel_count
                parcel_values_sum += parcel_values[i]
                total_parcel_mask = np.add(mask, total_parcel_mask)
            total_parcel_mask = (total_parcel_mask > 0)
            uniform_value = parcel_values_sum/pixel_count_sum
            uniform_value_map = np.add(total_parcel_mask*uniform_value, uniform_value_map)

            total_parcel_mask = (uniform_value_map > 0)
            sample = {'image':image, 'total_parcel_mask':total_parcel_mask,
                        'uniform_value_map': uniform_value_map}

            #sample = {'image': image,'masks': masks,
            #    'values':parcel_values,'polygons': polygons,
            #    'img_bbox': img_bbox}
            
        else:
            #for each mask turn to numpy array, flatten, and vstack
            masks = [torch.from_numpy( np.array(mask).flatten()) for mask in masks]
            parcel_masks = np.vstack(masks)

            parcel_values = torch.from_numpy(parcel_values)
            sample = {'image': image,'masks': parcel_masks,
                'values':parcel_values,'polygons': polygons,
                'img_bbox': img_bbox}

        return sample

def generate_dasymetric_map(polygons, img_bbox):
    fig, ax = plt.subplots()
    polygons.plot(ax=ax, column = 'TOTAL_MV1', alpha = 1, linewidth=3)
    dasy = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    return dasy

