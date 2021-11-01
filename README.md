# disaggregation

## Hennepin Dataset
/dataset/
This is for dowloading, organizing, and writing the hennepin dataset, 
configuration found within the script files

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
## Training
To train region aggregation models:
```/aggregation/train.py```

Training controls in ```/aggregation/config.py```

Generate visualizations/test statistics ```/aggregation/vis_gen.py``` (model selection in main)

WIP Testing script```/aggregation/test_script.py``` (model selection in main)

# TODO
    Configure models for various CNN architectures.(ResNet50)
    Probabalistic disaggregation
    Get rid of custom Collate, speed up training
