import os
import torch
import geopandas as gp
import glob
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import rasterio
import numpy as np

# This dataset class is for full inspection of the dataset, used for debugging
class HennepinDatasetFull(Dataset):
    def __init__(self, csv_path, shapefile_path, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.bbox_frame = pd.read_csv(csv_path)
        self.shp_path = shapefile_path
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.bbox_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #Bounding Box 
        row = self.bbox_frame.iloc[idx]
        row_bbox = (row['lat_min'], row['lon_min'],row['lat_max'], row['lon_max'])

        #Polygons
        gdf = gp.read_file(self.shp_path, bbox = row_bbox)
        #geometry = gdf['geometry']

        #Value
        value = gdf['TOTAL_MV1']

        #Image Bounding Boxw
        image_bbox = (row['lat_min'], row['lat_max'],row['lon_min'], row['lon_max'])
        
        #Image
        img_path = os.path.join(self.root_dir, str(int(row['lat_mid'])), str(int(row['lon_mid'])))
        pthList = sorted(glob.glob(img_path + '/*.tif'))

        #print(pthList)

        #print(img_path)
        #image = Image.open(pthList[0])
        raster = rasterio.open(pthList[0])
        array = raster.read()

        # Parcel Label
        try:
            label_raster = rasterio.open(pthList[2])
            parcel_label = label_raster.read()
            parcel_label = np.flip(parcel_label, 1)
        except IndexError:
            parcel_label = 0

        # Building Label
        try:
            label_raster = rasterio.open(pthList[1])
            building_label = label_raster.read()
            building_label = np.flip(building_label, 1)
        except IndexError:
            building_label = 0
        
        #Sample
        sample = {'image': array,'parcel_label': parcel_label,'building_label': building_label, 'raster': raster, 'bbox': row_bbox, 'img_bbox': image_bbox, 'geometry': gdf, 'value': value}

        return sample

class HennepinDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.bbox_frame = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.bbox_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #Bounding Box 
        row = self.bbox_frame.iloc[idx]
        row_bbox = (row['lat_min'], row['lon_min'],row['lat_max'], row['lon_max'])

        #Path
        img_path = os.path.join(self.root_dir, str(int(row['lat_mid'])), str(int(row['lon_mid'])))
        pthList = sorted(glob.glob(img_path + '/*.tif'))

        #Image
        raster = rasterio.open(pthList[0])
        array = raster.read()

        # Parcel Label
        try:
            label_raster = rasterio.open(pthList[2])
            parcel_label = label_raster.read()
            parcel_label = np.flip(parcel_label, 1)
        except IndexError:
            parcel_label = 0

        # Building Label
        try:
            label_raster = rasterio.open(pthList[1])
            building_label = label_raster.read()
            building_label = np.flip(building_label, 1)
        except IndexError:
            building_label = 0

        #Sample
        sample = {'image': array,'parcel_label': parcel_label,'building_label':building_label,'bbox': row_bbox}

        return sample