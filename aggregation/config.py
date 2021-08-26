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
cfg.data.root_dir = '/u/eag-d1/data/Hennepin/ver10/'
cfg.data.csv_path = '/u/eag-d1/data/Hennepin/ver10/hennepin_bbox.csv'
cfg.data.shp_path = '/u/eag-d1/data/Hennepin/ver10//hennepin.shp'
cfg.data.feather_path = '/u/eag-d1/data/Hennepin/ver10/'    

cfg.train = edict()

cfg.train.validation_split = 0.2         # percentage validation
cfg.train.test_split = 0.2               # percentage test
cfg.train.batch_size = 8
cfg.train.learning_rate = 5e-4           # initial learning rate
cfg.train.l2_reg = 1e-6
cfg.train.lr_decay = 0.9
cfg.train.lr_decay_every = 3
cfg.train.shuffle = True                 # shuffle training samples
cfg.train.num_epochs = 200                # number of training epochs  ...
cfg.train.num_workers = 8                # workers for data loading
cfg.train.device_ids = [0,2]                 # Train on two GPUs? Set True for blackbird

cfg.train.loss_weight =  [0.23498031, 0.0815268,  2.53768995, 1.14580294]           # loss weights used during training

cfg.train.out_dir = './outputs/test'        # [3] fix labels, train w/ loss weights

# evaluation settings
cfg.data.eval_mode = 'test'              # evaluation split. options: 'val', 'test'