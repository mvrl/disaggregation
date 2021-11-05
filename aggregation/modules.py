import pytorch_lightning as pl
import torch 
import util
from models import unet
import torch.nn as nn
import torch.distributions as dist


class RALModule(pl.LightningModule):
    def __init__(self,use_pretrained):
        super().__init__()
        self.unet = unet.UNet(in_channels=3, out_channels=1)

        # This wont work, since the pretrained unet has 2 output channels
        if(use_pretrained):
            pretrained_state = torch.load('/u/eag-d1/data/Hennepin/model_checkpoints/building_seg_pretrained.pth')

            model_state = self.unet.state_dict()
            pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
            model_state.update(pretrained_state)
            self.unet.load_state_dict(model_state)
        
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
            pretrained_state = torch.load('/u/eag-d1/data/Hennepin/model_checkpoints/building_seg_pretrained.pth')

            model_state = self.unet.state_dict()
            pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
            model_state.update(pretrained_state)
            self.unet.load_state_dict(model_state)

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
        output = torch.flatten(output, start_dim=1)
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

class ProbabalisticModule(pl.LightningModule):
    def __init__(self,use_pretrained):
        super().__init__()
        self.unet = unet.UNet(in_channels=3, out_channels=2)

        # This wont work, since the pretrained unet has 2 output channels
        if(use_pretrained):
            pretrained_state = torch.load('/u/eag-d1/data/Hennepin/model_checkpoints/building_seg_pretrained.pth')

            model_state = self.unet.state_dict()
            pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
            model_state.update(pretrained_state)
            self.unet.load_state_dict(model_state)

        self.softplus = nn.Softplus()
        self.agg = util.regionAgg_layer()

    def get_valOut(self, x):
        means, stds = self(x)
        return means.unsqueeze(1)

    def forward(self, x):
        x = self.unet(x)
        x = self.softplus(x)
        means = x[:,0]
        stds = x[:,1]
        return means, stds

    def pred_Out(self, x, masks):
        means, stds = self(x)
        print(stds)
        output = torch.flatten(means, start_dim=1)
        estimated_values = self.agg(output, masks)
        return estimated_values

    def shared_step(self, batch):
        image, parcel_masks, parcel_values = batch

        means, stds = self(image)

        gauss = dist.Normal(means, stds)
        sample = gauss.rsample()
        output = torch.flatten(sample, start_dim=1)
        
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

class GaussModule(pl.LightningModule):
    def __init__(self,use_pretrained):
        super().__init__()
        self.unet = unet.UNet(in_channels=3, out_channels=2)

        # This wont work, since the pretrained unet has 2 output channels
        if(use_pretrained):
            pretrained_state = torch.load('/u/eag-d1/data/Hennepin/model_checkpoints/building_seg_pretrained.pth')

            model_state = self.unet.state_dict()
            pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
            model_state.update(pretrained_state)
            self.unet.load_state_dict(model_state)
        
        self.softplus = nn.Softplus()
        self.agg = util.regionAgg_layer()
        self.gauss_agg = util.gaussAgg_layer()

        # For the gaussian distribution summations, how should I consider the value prediction at inference time?

    def forward(self, x):
        x = self.unet(x)
        x = self.softplus(x)
        means = x[:,0]
        vars = x[:,1] 
        means = torch.flatten(means, start_dim=1)
        vars = torch.flatten(vars, start_dim=1)
        return means, vars

    def get_valOut(self, x):
        x = self.unet(x)
        x = self.softplus(x)
        means = x[:,0]
        vars = x[:,1] 
        return means.unsqueeze(1)

    def pred_Out(self, x, masks):
         means, vars = self(x)
         estimated_values = self.agg(means, masks)
         return estimated_values

    def shared_step(self, batch):
        image, parcel_masks, parcel_values = batch

        means, vars = self(image)
        
        #take cnn output and parcel masks(Aggregation Matrix M)
        means, vars = self.gauss_agg(means, vars, parcel_masks)
        loss = util.gaussLoss(means, vars, parcel_values)
        
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