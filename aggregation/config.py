# This file contains configuration for training and evaluation

from easydict import EasyDict as edict

cfg = edict()

cfg.use_existing = True                 # use existing method for calculating loss (True) or Ben's blocking method (False)

## MODEL
cfg.model = edict()
cfg.model.name = 'unet'                  # 'unet', 'unet_normalize', 'hr_net'
cfg.model.reduction = 4                  # reduction factor for number of feature maps. 4 means a network with 1/4 feature maps
cfg.model.out_channels = 4   


## DATA
cfg.data = edict()
cfg.data.name = 'hennepin'               # only 'hennepin' for now

cfg.data.cutout_size = (302, 302)        # final image size. Not implemented yet
cfg.data.root_dir = '/u/eag-d1/data/Hennepin/new_area/'
cfg.data.sample_mode = 'uniform'


cfg.train = edict()
cfg.mode = 'train'

cfg.experiment_name = 'uniform_notpre'

cfg.train.use_pretrained = False
cfg.train.model = 'uniform'
cfg.train.validation_split = 0.2         # percentage validation
cfg.train.test_split = 0.2               # percentage test
cfg.train.batch_size = 16
cfg.train.shuffle = True                 # shuffle training samples
cfg.train.num_epochs = 150               # number of training epochs  ...
cfg.train.num_workers = 8                # workers for data loading
cfg.train.device_ids = [0]               # Train on two GPUs? Set True for blackbird
