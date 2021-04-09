import pytorch_lightning as pl
import torch 
import numpy as np
import pandas as pd

import data_factory
from torch.utils.data import Dataset, DataLoader

from models import unet
from config import cfg
import util

class aggregationModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.unet = unet.Unet(in_channels=3, out_channels=1)
        self.agg = util.regionAgg_layer(1*224*224)


    def forward(self, x):
        x = self.unet(x)
        x = torch.flatten(x, start_dim=1)
        return x

    def training_step(self, batch, batch_idx):

        image, value_map, parcel_masks, parcel_values = batch

        output = self(image)

        #take cnn output and parcel masks(Aggregation Matrix M)
        self.agg(output, parcel_masks)

        loss = util.regionAgg_loss(output, target)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



if __name__ == '__main__':

    this_dataset = data_factory.dataset_hennepin('train','/u/eag-d1/data/Hennepin/ver3/',
    '/u/eag-d1/data/Hennepin/ver3/hennepin_bbox.csv',
    '/u/pop-d1/grad/cgar222/Projects/disaggregation/dataset/hennepin_county_parcels/hennepin_county_parcels.shp')

    torch.manual_seed(0)

    train_size = int( np.floor( len(this_dataset) * (1-cfg.train.validation_split-cfg.train.test_split) ) )
    val_size = int( np.floor( len(this_dataset) * cfg.train.validation_split) )
    test_size = int( np.floor( len(this_dataset) * cfg.train.test_split ) )

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(this_dataset, [train_size, val_size, test_size])\


    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=cfg.train.shuffle,
                             num_workers=cfg.train.num_workers)

    model = aggregationModule()
    trainer = pl.Trainer()
    trainer.fit(model, train_loader)
    print('what')