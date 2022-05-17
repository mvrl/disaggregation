# This file contains configuration for training and evaluation

from easydict import EasyDict as edict

cfg = edict()

cfg.use_existing = True                 # use existing method for calculating loss (True) or Ben's blocking method (False)

## CNN MODEL
cfg.model = edict()
cfg.model.name = 'unet'                  # 'unet', 'unet_normalize', 'hr_net'
cfg.model.reduction = 4                  # reduction factor for number of feature maps. 4 means a network with 1/4 feature maps
cfg.model.out_channels = 4   


## DATA
cfg.data = edict()

cfg.data.name = 'hennepin'               # 'hennepin' , 'cifar'

cfg.data.hennepin = edict()
#Hennepin specific settings
cfg.data.hennepin.root_dir = '/u/eag-d1/data/Hennepin/new_area_large/' #512x512 patch size
                    # /u/eag-d1/data/Hennepin/new_area/ 302x302 patch size
cfg.data.hennepin.sample_mode = ''


cfg.data.cutout_size = (302, 302)        # Used in visualizations


cfg.experiment_name = 'ral_combined'

cfg.train = edict()
cfg.mode = 'train'
cfg.train.model = 'ral'

cfg.train.use_pretrained = False

cfg.train.patience = 100

cfg.train.lam = 1e7
cfg.train.num_samples = 100  #only with LogSample model
 
cfg.train.validation_split = 0.1         # percentage validation
cfg.train.test_split = 0.1               # percentage test
cfg.train.batch_size = 8
cfg.train.shuffle = True                 # shuffle training samples
cfg.train.num_epochs = 300              # number of training epochs  ...
cfg.train.num_workers = 4                # workers for data loading
cfg.train.device_ids = [1]               # Train on two GPUs? Set True for blackbird

