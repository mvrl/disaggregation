import os
import csv
import time
import glob
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import urllib.request
from multiprocessing import Pool

import utils
#from dataset import vec_XY, all_locations

def download(item):
  url = item[0]
  fn = item[1]

  if not os.path.exists(fn):
    utils.ensure_dir(fn)
    try:
      urllib.request.urlretrieve(url, fn)
    except:
      print('error while retrieving {:}'.format(url))
  else:
    pass
    #print('Skipped ', fn.split('/')[-1])


if __name__ == "__main__":

  #image size
  image_width = 302 * 2
  delta_X = image_width / 2

  #ESPG:26915 - meters
  X_min_bound = 440000  
  X_max_bound = 473000
  Y_min_bound = 4974000
  Y_max_bound = 4988000

  #Mesh
  X = np.arange( X_min_bound, X_max_bound, step = image_width )
  Y = np.arange( Y_min_bound, Y_max_bound, step = image_width )
  mesh = np.array(np.meshgrid(X,Y))

  combinations = mesh.T.reshape(-1, 2)

  locations_to_download = np.unique(combinations, axis=0)

  #Make some crummy bounding boxes, TODO make these good shapes in meters
  df = pd.DataFrame(
      np.concatenate([locations_to_download + x for x in [-delta_X, 0, delta_X]],
                     axis=1),
      columns=('lat_min', 'lon_min', 'lat_mid', 'lon_mid', 'lat_max',
               'lon_max'))

  #naip_dir = '/u/eag-d1/scratch/ebird/naip/'

  #print(df.head())

  df.to_csv('hennepin_bbox.csv')

  if __name__ == '__main__':
    #utils.ensure_dir(naip_dir)

    #template_url = 'http://kyraster.ky.gov/arcgis/services/ImageServices/Ky_NAIP_2014_1M/ImageServer/WMSServer?request=GetMap&service=WMS&layers=0&CRS=EPSG:4326&BBOX={:},{:},{:},{:}&width=302&height=302&format=image/tif'
    template_url = 'https://gis.hennepin.us/arcgis/services/Imagery/UTM_Aerial_2020/MapServer/WMSServer?version=1.3.0&&service=WMS&request=GetMap&&styles=&layers=0&CRS=EPSG:26915&BBOX={:},{:},{:},{:}&width=302&height=302&format=image/jpg'

    print('Preparing jobs')

    jobs = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
      #if idx % 10000 == 0: print('%d / %d' % (idx, len(df)))

      image_url = template_url.format(row['lat_min'], row['lon_min'],
                                      row['lat_max'], row['lon_max'])
                                      
      #print(image_url)
      out_file = 'image_set/' + "{:}/{:}/{:}_{:}.tiff".format(
          int(row['lat_mid']), int(row['lon_mid']),
          row['lat_mid'], row['lon_mid'])
      jobs.append([image_url, out_file])

  print('Started downloading')
  start = time.time()
  p = Pool(10)

  for fn in tqdm(p.imap_unordered(download, jobs), total=len(jobs)):
    pass

  end = time.time()