# Requirements
import pytorch_lightning as pl
import torch 

#Local
from models import unet
import util
from config import cfg

# Testing, small imports
from pytorch_lightning.callbacks import ModelCheckpoint
import time
import torch.nn as nn


'''
    TBD:
        -Move model selection, data-selection, epochs, hyper-parameters all to config
        - Add a model for CIFAR?
'''
class OnexOneAggregationModule(pl.LightningModule):
    def __init__(self, use_pretrained):
        super().__init__()
        self.unet = unet.UNet(in_channels=3, out_channels=2)

        if(use_pretrained):
            state_dict = torch.load('/u/eag-d1/data/Hennepin/model_checkpoints/building_seg_pretrained.pth')

            self.unet.load_state_dict(state_dict)

            #for param in self.unet.parameters():
            #    param.requires_grad = False

        self.conv = nn.Conv2d(2,1,kernel_size = 1, padding = 0)

        self.softplus = nn.Softplus()
        self.agg = util.regionAgg_layer()

    def forward(self, x):
        x = self.valOut(x)
        x = torch.flatten(x, start_dim=1)
        return x

    def cnnOut(self, x):
        x = self.unet(x)
        return x

    def valOut(self, x):
        x = self.cnnOut(x)
        x = self.conv(x)
        x = self.softplus(x)
        return x

    def pred_Out(self, x, masks):
         output = self(x)
         estimated_values = self.agg(output, masks)
         return estimated_values
        
    def shared_step(self, batch):
        image, parcel_masks, parcel_values = batch

        output = self(image)
        
        #take cnn output and parcel masks(Aggregation Matrix M)
        estimated_values = self.agg(output, parcel_masks)
        loss = util.MAE(estimated_values, parcel_values)
        
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
        self.unet = unet.UNet(in_channels=3, out_channels=1)

        # This wont work, since the pretrained unet has 2 output channels
        if(use_pretrained):
            state_dict = torch.load('/u/eag-d1/data/Hennepin/model_checkpoints/building_seg_pretrained.pth')
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

    def pred_Out(self, x, masks):
         output = self(x)
         estimated_values = self.agg(output, masks)
         return estimated_values

    def shared_step(self, batch):
        image, parcel_masks, parcel_values = batch

        output = self(image)
        
        #take cnn output and parcel masks(Aggregation Matrix M)
        estimated_values = self.agg(output, parcel_masks)
        loss = util.MAE(estimated_values, parcel_values)
        
        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('train_loss', output['loss'], on_epoch = True)
        return output['loss']

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('val_loss', output['loss'], on_epoch = True)
        return output['loss']

    def test_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('test_loss', output['loss'], on_epoch = True)
        return output['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class UniformModule(pl.LightningModule):
    def __init__(self,use_pretrained):
        super().__init__()
        self.unet = unet.UNet(in_channels=3, out_channels=1)

        # This wont work, since the pretrained unet has 2 output channels
        if(use_pretrained):
            state_dict = torch.load('/u/eag-d1/data/Hennepin/model_checkpoints/building_seg_pretrained.pth')
            self.unet.load_state_dict(state_dict)

            for param in self.unet.parameters():
                param.requires_grad = False

        self.loss = nn.L1Loss()
        self.agg = util.regionAgg_layer()

    def forward(self, x):
        x = self.unet(x)
        return x

    def cnnOutput(self, x):
        x = self.unet(x)
        return x

    def get_valOut(self, x):
        x = self.unet(x)
        return x

    def pred_Out(self, x, masks):
         output = self(x)
         estimated_values = self.agg(output, masks)
         return estimated_values

    def shared_step(self, batch):
        output = self(batch['image']).squeeze(1)
        output = torch.mul(output,batch['total_parcel_mask'])
        loss = self.loss(output,batch['uniform_value_map'])
        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('train_loss', output['loss'], on_epoch = True)
        return output['loss']

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('val_loss', output['loss'], on_epoch = True)
        return output['loss']

    def test_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('test_loss', output['loss'], on_epoch = True)
        return output['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def chooseModel(model_name = cfg.model):

    if model_name == "end2end":
        model = End2EndAggregationModule(use_pretrained=False)
    if model_name == "1x1":
        model = OnexOneAggregationModule(use_pretrained=True)
    if model_name == "uniform":
        model = OnexOneAggregationModule(use_pretrained=True)
    return model

if __name__ == '__main__':


    train_loader, val_loader, test_loader = util.make_loaders(uniform=True)

    #Init ModelCheckpoint callback, monitoring 'val_loss'
    ckpt_monitors = (
        ModelCheckpoint(monitor='val_loss')
        )
    ckpt_monitors = ()

    model = UniformModule(use_pretrained=False)
    #trainer = pl.Trainer(gpus=[1], max_epochs = cfg.train.num_epochs, checkpoint_callback=True, callbacks=[*ckpt_monitors])
    trainer = pl.Trainer(gpus=[0], checkpoint_callback=True, callbacks=[*ckpt_monitors])
    t0 = time.time()
    trainer.fit(model, train_loader, val_loader)
    t1 = time.time()
    print('Training completed in', t1-t0, 'secs')
    print('Training completed in',(t1-t0)/60, 'mins')
    
