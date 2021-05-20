# This file contains configuration for training and evaluation

from easydict import EasyDict as edict

cfg = edict()

## MODEL
cfg.model = edict()
cfg.model.name = 'unet'                  # 'unet', 'unet_normalize', 'hr_net'
cfg.model.reduction = 3                  # reduction factor for number of feature maps. 4 means a network with 1/4 feature maps
cfg.model.out_channels = 3   


## DATA
cfg.data = edict()
cfg.data.name = 'hennepin'               # only 'hennepin' for now

cfg.data.cutout_size = (302, 302)        # final image size. Not implemented yet
cfg.data.root_dir = '/u/eag-d1/data/Hennepin/ver1/'
cfg.data.csv_path = '/u/eag-d1/data/Hennepin/ver1/hennepin_bbox.csv'

cfg.train = edict()

cfg.train.validation_split = 0.2         # percentage validation
cfg.train.test_split = 0.1               # percentage test
cfg.train.batch_size = 16
cfg.train.learning_rate = 5e-4           # initial learning rate
cfg.train.l2_reg = 1e-6
cfg.train.lr_decay = 0.9
cfg.train.lr_decay_every = 3
cfg.train.shuffle = True                 # shuffle training samples
cfg.train.num_epochs = 50                # number of training epochs  ...
cfg.train.num_workers = 4                # workers for data loading
cfg.train.device_ids = [0,2]                 # Train on two GPUs? Set True for blackbird

cfg.train.loss_weight =  [0.23498031, 0.0815268, 1.14580294]           # loss weights used during training

cfg.train.out_dir = './outputs/segpretrain'        # [3] fix labels, train w/ loss weights

# evaluation settings
cfg.data.eval_mode = 'test'              # evaluation split. options: 'val', 'test'