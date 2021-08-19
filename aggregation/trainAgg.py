import pytorch_lightning as pl
import torch 

from models import unet
import util

from collections import OrderedDict
import torch.nn as nn
from torchvision.utils import save_image
from pytorch_lightning.callbacks import ModelCheckpoint
import time
import torch.nn as nn
from config import cfg


class aggregationModule(pl.LightningModule):
    def __init__(self, use_pretrained, use_existing=True):
        super().__init__()
        self.unet = unet.UNet(in_channels=3, out_channels=2)

        if(use_pretrained):
            state_dict = torch.load('/u/eag-d1/data/Hennepin/model_checkpoints/building_seg_pretrained.pth')
            #Removing dictionary elements from nn.dataParrelel
            #new_state_dict = OrderedDict()
            #for k, v in state_dict.items():
            #    name = k[7:] # remove module.
            #    new_state_dict[name] = v

            self.unet.load_state_dict(state_dict)

            for param in self.unet.parameters():
                param.requires_grad = False

        self.conv = nn.Conv2d(2,1,kernel_size = 1, padding = 0)

        self.softplus = nn.Softplus()
        self.agg = util.regionAgg_layer()
        self.criterion = nn.MSELoss()

        self.use_existing = use_existing

    def forward(self, x):
        x = self.unet(x)
        x = self.conv(x)
        x = self.softplus(x)
        x = torch.flatten(x, start_dim=1)
        return x

    def cnnOutput(self, x):
        x = self.unet(x)
        return x

    def get_valOut(self, x):
        x = self.unet(x)
        x = self.conv(x)
        x = self.softplus(x)
        return x

    def shared_step(self, batch):
        image, parcel_masks, parcel_values = batch

        output = self(image)
        
        #take cnn output and parcel masks(Aggregation Matrix M)
        estimated_values = self.agg(output, parcel_masks, self.use_existing)
        loss = util.MSE(estimated_values, parcel_values, self.use_existing)
        
        return {'loss': loss}


    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('train_loss', output['loss'])
        return output['loss']

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('val_loss', output['loss'])
        return output['loss']

    def test_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('test_loss', output['loss'])
        return output['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
class End2EndAggregationModule(pl.LightningModule):
    def __init__(self,use_pretrained):
        super().__init__()
        self.unet = unet.Unet(in_channels=3, out_channels=1)

        # This wont work, since the pretrained unet has 2 output channels
        if(use_pretrained):
            state_dict = torch.load('/u/pop-d1/grad/cgar222/Projects/disaggregation/building_segmentation/outputs/segpretrain2/model_dict.pth')
            self.unet.load_state_dict(state_dict)

            for param in self.unet.parameters():
                param.requires_grad = False
        
        self.softplus = nn.Softplus()
        self.agg = util.regionAgg_layer()

    def forward(self, x):
        x = self.unet(x)
        x = self.softplus(x)
        x = torch.flatten(x, start_dim=1)
        return x

    def cnnOutput(self, x):
        x = self.unet(x)
        return x

    def get_valOut(self, x):
        x = self.unet(x)
        x = self.softplus(x)
        return x

    def training_step(self, batch, batch_idx):

        image, parcel_masks, parcel_values = batch

        output = self(image)

        #take cnn output and parcel masks(Aggregation Matrix M)
        estimated_values = self.agg(output, parcel_masks)

        loss = util.MSE(estimated_values, parcel_values)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):

        image, parcel_masks, parcel_values = batch

        output = self(image)

        estimated_values = self.agg(output, parcel_masks)

        loss = util.MSE(estimated_values, parcel_values)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):

        image, parcel_masks, parcel_values = batch

        output = self(image)

        estimated_values = self.agg(output, parcel_masks)

        loss = util.MAE(estimated_values, parcel_values)
        self.log('test_loss', loss)
        return loss


if __name__ == '__main__':
    use_existing = cfg.use_existing
    train_loader, val_loader, test_loader = util.make_loaders()

    #Init ModelCheckpoint callback, monitoring 'val_loss'
    ckpt_monitors = (
        ModelCheckpoint(monitor='val_loss')
        )
    ckpt_monitors = ()

    model = aggregationModule(use_pretrained=False, use_existing=use_existing)
    trainer = pl.Trainer(gpus=[0], max_epochs = cfg.train.num_epochs, checkpoint_callback=False, callbacks=[*ckpt_monitors])
    t0 = time.time()
    trainer.fit(model, train_loader, val_loader)
    t1 = time.time()
    print('Training completed in', t1-t0, 'secs')
    print('Training completed in',(t1-t0)/60, 'mins')
    
