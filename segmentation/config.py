"""Configuration for the building-segmentation pre-training step.

Run this BEFORE the aggregation pipeline if you want use_pretrained=True. It produces
$SEG_OUT_DIR/building_seg_pretrained.pth, which aggregation/modules.py loads via
the PRETRAINED_CKPT env var (or its default that points back here).
"""

import os
import sys

import torch
from easydict import EasyDict as edict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

sys.path.insert(0, REPO_ROOT)
import paths  # noqa: F401
sys.path.pop(0)

cfg = edict()

## MODEL
cfg.model = edict()
cfg.model.name = 'unet'                  # 'unet', 'nested_unet'
cfg.model.reduction = 2
cfg.model.out_channels = 2

## DATA
cfg.data = edict()
cfg.data.name = 'hennepin'

cfg.data.cutout_size = (302, 302)
# `ver8/` layout: per-tile subdirs keyed by lat_mid/lon_mid containing
# building_mask.tif, parcel_boundary.tif, parcel_mask.tif, <lat>_<lon>.tif,
# parcel_value.tif (and optionally parcels.shp). NOT in the Box archives;
# point HENNEPIN_VER8_ROOT at the cluster copy when running on the server.
cfg.data.root_dir = os.environ.get(
    'HENNEPIN_VER8_ROOT', os.path.join(REPO_ROOT, 'data', 'hennepin_ver8')
)
cfg.data.csv_path = os.environ.get(
    'HENNEPIN_VER8_CSV',
    os.path.join(REPO_ROOT, 'dataset_generation', 'hennepin_bbox.csv'),
)
cfg.data.eval_mode = 'test'

## TRAIN
cfg.train = edict()
cfg.train.validation_split = 0.2
cfg.train.test_split = 0.1
cfg.train.batch_size = 16
cfg.train.learning_rate = 5e-4
cfg.train.l2_reg = 1e-6
cfg.train.lr_decay = 0.9
cfg.train.lr_decay_every = 3
cfg.train.shuffle = True
cfg.train.num_epochs = 50
cfg.train.num_workers = 4
cfg.train.device_ids = [0]               # used only when CUDA is available
cfg.train.device = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg.train.loss_weight = []

cfg.train.out_dir = os.environ.get(
    'SEG_OUT_DIR', os.path.join(REPO_ROOT, 'segmentation', 'outputs', 'buildsegpretrain')
)
