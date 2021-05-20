import pytorch_lightning as pl
import torch 
import numpy as np
import pandas as pd

import data_factory
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

from models import unet
from config import cfg
import util

from collections import OrderedDict
import torch.nn as nn
from torchvision.utils import save_image


class aggregationModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.unet = unet.Unet(in_channels=3, out_channels=3)

        state_dict = torch.load('/u/pop-d1/grad/cgar222/Projects/disaggregation/building_segmentation/outputs/segpretrain/model_dict.pth')
        #Removing dictionary elements from nn.dataParrelel
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove module.
            new_state_dict[name] = v

        self.unet.load_state_dict(new_state_dict)

        for param in self.unet.parameters():
            param.requires_grad = False

        self.conv = nn.Conv2d(3,1,kernel_size = 1, padding = 0)
        self.softplus = nn.Softplus()

        # I will need a new dataloader... and fix the labels for the value maps


    def forward(self, x):
        #self.unet.eval()
        #with torch.no_grad():
        x = self.unet(x)

        save_image(x[0], 'img1.png')
        x = self.conv(x)
        x = self.softplus(x)

        save_image(x[0], 'img2.png')
        
        #print(x[0])

        x = torch.flatten(x, start_dim=1)
        return x

    def cnnOutput(self, x):
        x = self.unet(x)
        return x

    def training_step(self, batch, batch_idx):

        image, parcel_masks, parcel_values = batch

        output = self(image)

        #print(parcel_values)

        #take cnn output and parcel masks(Aggregation Matrix M)
        estimated_values = self.agg(output, parcel_masks)

        loss = util.regionAgg_loss(estimated_values, parcel_values)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# Minibatch creation for variable size targets in Hennepin Dataset
def my_collate(batch):

    #Masks and values are in lists        
    mask = [item[1] for item in batch]
    value = [item[2] for item in batch]


    image = [item[0].unsqueeze(0) for item in batch]
    image = torch.cat(image)

    return image, mask, value
    



if __name__ == '__main__':

    this_dataset = data_factory.dataset_hennepin('train','/u/eag-d1/data/Hennepin/ver7/',
    '/u/eag-d1/data/Hennepin/ver7/hennepin_bbox.csv',
    '/u/pop-d1/grad/cgar222/Projects/disaggregation/dataset/hennepin_county_parcels/hennepin_county_parcels.shp')

    torch.manual_seed(0)

    #train_size = int( np.floor( len(this_dataset) * (1-cfg.train.validation_split-cfg.train.test_split) ) )
    #val_size = int( np.floor( len(this_dataset) * cfg.train.validation_split) )
    #test_size = int( np.floor( len(this_dataset) * cfg.train.test_split ) )

    #train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(this_dataset, [train_size, val_size, test_size])\


    train_loader = DataLoader(this_dataset, batch_size=16, shuffle=cfg.train.shuffle, collate_fn = my_collate,
                             num_workers=cfg.train.num_workers)

    model = aggregationModule()
    trainer = pl.Trainer(gpus='0')
    trainer.fit(model, train_loader)
    print('what')
