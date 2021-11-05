# disaggregation

## Training
Training controls in ```/aggregation/config.py```

To train region aggregation models:
```/aggregation/train.py```
    - the best weights will be saved to /results/config.experiment_name/

Generate visualizations/test statistics ```/aggregation/vis.py``` 
    - will generate 100 test images saved to /results/config.experiment_name/visualizations/{}.png

Testing script```/aggregation/test.py```
    - this will save pkl files of predicted and actual values
    - will save test statistics to /results/config.experiment_name/stats.txt

# When running an experiment
Be sure to set the config for these when running a new experiment:
```cfg.data.sample_mode = ''``` certian models require a different sample mode for the dataset. In general leave blank.
```cfg.experiment_name = 'experiment_name_goes_here'```
```cfg.train.model = 'model_name' ```
```cfg.train.use_pretrained = Boolean```
model selects a module in modules.py.

# TODO
    Configure models for various CNN architectures.(ResNet50, Efficient UNet)
    Probabalistic disaggregation
    Poisson Distribution
    Visualization of Uncertainty/Variance
    Higher res?

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
## Segmentation
Included here is Usman's code for training a segmentation network on Hennepin dataset.
We use this to generate pre-trained weights for the value estimation step.

The model ckpts are stored in ```/outputs/```