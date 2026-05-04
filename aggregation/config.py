"""Configuration for training and evaluation of the aggregation models."""

import os
import sys

from easydict import EasyDict as edict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Load <repo>/paths.env into os.environ (does not override real exported env vars).
sys.path.insert(0, REPO_ROOT)
import paths  # noqa: F401 — side effect: populates os.environ from paths.env
sys.path.pop(0)

cfg = edict()

cfg.use_existing = True                  # use existing method for calculating loss (True) or Ben's blocking method (False)

## CNN MODEL
cfg.model = edict()
cfg.model.name = 'unet'                  # 'unet', 'unet_normalize', 'hr_net'
cfg.model.reduction = 4                  # reduction factor for number of feature maps. 4 means a network with 1/4 feature maps
cfg.model.out_channels = 4

## DATA
cfg.data = edict()
cfg.data.name = 'hennepin'

cfg.data.hennepin = edict()
# Root directory containing the unpacked Hennepin archives. Override with HENNEPIN_DATA_ROOT.
# Expected layout under the root:
#   <root>/1m_302px/{imgs,masks,vals.pkl}                  (uncombined parcels)
#   <root>/1m_302px_region_combined/{imgs,masks,vals.pkl}  (combined parcels)
# scripts/setup_hennepin.sh produces this layout from the Box archives.
cfg.data.hennepin.root_dir = os.environ.get(
    'HENNEPIN_DATA_ROOT', os.path.join(REPO_ROOT, 'data', 'hennepin')
)
cfg.data.hennepin.uncombined_subdir = os.environ.get('HENNEPIN_UNCOMBINED_SUBDIR', '1m_302px')
cfg.data.hennepin.combined_subdir = os.environ.get('HENNEPIN_COMBINED_SUBDIR', '1m_302px_region_combined')
cfg.data.hennepin.sample_mode = ''

cfg.data.cutout_size = (302, 302)        # used in visualizations

cfg.experiment_name = os.environ.get('EXPERIMENT_NAME', 'gauss_covariance')

cfg.train = edict()
cfg.mode = 'train'
cfg.train.model = os.environ.get('TRAIN_MODEL', 'gauss')   # 'ral' | 'uniform' | 'rsample' | 'gauss' | 'logsample'
cfg.train.use_pretrained = os.environ.get('USE_PRETRAINED', '0') == '1'

# Path to the building-segmentation pretrained checkpoint. Override with PRETRAINED_CKPT.
# Produced by segmentation/train.py at $SEG_OUT_DIR/building_seg_pretrained.pth.
cfg.train.pretrained_ckpt = os.environ.get(
    'PRETRAINED_CKPT',
    os.path.join(REPO_ROOT, 'segmentation', 'outputs', 'buildsegpretrain', 'building_seg_pretrained.pth'),
)

cfg.train.patience = int(os.environ.get('PATIENCE', '100'))

cfg.train.lam = 1e7
cfg.train.num_samples = 1                # only used by the LogSample model

cfg.train.validation_split = 0.1
cfg.train.test_split = 0.1
cfg.train.batch_size = int(os.environ.get('BATCH_SIZE', '8'))
cfg.train.shuffle = True
cfg.train.num_epochs = int(os.environ.get('NUM_EPOCHS', '300'))
cfg.train.num_workers = int(os.environ.get('NUM_WORKERS', '4'))

# PyTorch Lightning 2.x trainer knobs.
cfg.train.accelerator = os.environ.get('PL_ACCELERATOR', 'auto')   # 'auto' | 'cpu' | 'gpu' | 'mps'
cfg.train.devices = int(os.environ.get('PL_DEVICES', '1'))
