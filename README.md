# disaggregation




## Hennepin Dataset
Download Parcel Zipfile
```
cd dataset
chmod u+x request_zip.sh
./request_zip.sh
```
Download Images
```
cd dataset
python satellite_gather.py --path /u/eag-d1/data/Hennepin/ver1/ --gsd 1
```
Create Labels - Slow!
```
cd dataset
python generate_labels.py --path /u/eag-d1/data/Hennepin/ver1/ --gsd 1
```
## Building Segmentation
