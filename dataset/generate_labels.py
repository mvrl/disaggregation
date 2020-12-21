import rasterize
import pandas as pd
import os
from tqdm import tqdm

csv_path = './hennepin_bbox.csv'

root_dir = './image_set'

bbox_df = pd.read_csv(csv_path)

if __name__ == "__main__":

    # Loop through CSV


    with tqdm(total = len(bbox_df)) as pbar:
        for index,row in bbox_df.iterrows():

            label_path = os.path.join(root_dir, str(int(row['lat_mid'])), str(int(row['lon_mid'])))

            label_path = os.path.join(label_path, 'label.tif') # need a better label filename 

            image_bbox = (row['lat_min'], row['lat_max'],row['lon_min'], row['lon_max'])

            #print(label_path)


            # For each BBOX generate raster and filepath
            rasterize.raster(bbox = image_bbox, fn= label_path)

            pbar.update(1)
