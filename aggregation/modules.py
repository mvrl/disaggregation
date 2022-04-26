from statistics import variance
import pytorch_lightning as pl
import torch 
import util
from models import unet
import torch.nn as nn
import torch.distributions as dist
from config import cfg

# Linear Basline
class RALModule(pl.LightningModule):
    def __init__(self,use_pretrained):
        super().__init__()
        self.unet = unet.UNet(in_channels=3, out_channels=1)

        if(use_pretrained):
            pretrained_state = torch.load('/u/eag-d1/data/Hennepin/model_checkpoints/building_seg_pretrained.pth')

            model_state = self.unet.state_dict()
            pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
            model_state.update(pretrained_state)
            self.unet.load_state_dict(model_state)

        self.loss_torch = torch.nn.MSELoss()

    def forward(self, x):
        x = self.unet(x)
        return x # B x 1 x H x W

    def value_predictions(self, batch):
        image, masks, values = batch['image'], batch['masks'], batch['values']

        vals = self(image)
        # Shape: B X 1 X H X W

        vals = torch.flatten(vals, start_dim=1)
        # Shape: B X 1 X HW

        masks = torch.flatten(masks,start_dim = 2)
        masks = torch.swapdims(masks,2,1)
        # Shape: B X HW X 100 'max 100 parcels in a sample"
        
        #Aggregate
        vals_sums = torch.matmul(vals.unsqueeze(1).float(), masks.float()).squeeze(1)
        #Shape: B X 100

        #We need to ignore the zeroes
        indices = vals_sums.nonzero(as_tuple=True)
        vals_sums = vals_sums[indices]
        values = values[indices]
        #Shape: num_parcels (IN ALL OF BATCH)

        return vals_sums, values

    def shared_step(self, batch):
        image, masks, values = batch['image'], batch['masks'], batch['values']

        output = self(image)
        output = torch.flatten(output, start_dim=2)

        masks = torch.flatten(masks,start_dim = 2)
        masks = torch.swapdims(masks,2,1)
    
        region_sums_yhat = torch.matmul(output.float(), masks.float()).squeeze(1)

        squares = torch.square(values.float()-region_sums_yhat)
        loss = torch.sum(squares, dim=1).mean()
        
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

    def forward(self, x):
        x = self.unet(x)
        return x # B x 1 x H x W

    def cnnOutput(self, x):
        x = self.unet(x)
        return x

    def value_predictions(self, batch):
        image, masks, values = batch['image'], batch['masks'], batch['values']

        vals = self(image)
        # Shape: B X 1 X H X W

        vals = torch.flatten(vals, start_dim=1)
        # Shape: B X 1 X HW

        masks = torch.flatten(masks,start_dim = 2)
        masks = torch.swapdims(masks,2,1)
        # Shape: B X HW X 100 'max 100 parcels in a sample"
        
        #Aggregate
        vals_sums = torch.matmul(vals.unsqueeze(1).float(), masks.float()).squeeze(1)
        #Shape: B X 100

        #We need to ignore the zeroes
        indices = vals_sums.nonzero(as_tuple=True)
        vals_sums = vals_sums[indices]
        values = values[indices]
        #Shape: num_parcels (IN ALL OF BATCH)

        # est values, true values
        return vals_sums, values

    def shared_step(self, batch):
        output = self(batch['image']).squeeze(1)
        output = torch.mul(output,batch['total_parcel_mask'])
        loss = self.loss(output,batch['uniform_value_map'])
        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('train_loss', output['loss'], on_epoch = True, batch_size=cfg.train.batch_size)
        return output['loss']

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('val_loss', output['loss'], on_epoch = True, batch_size=cfg.train.batch_size)
        return output['loss']

    def test_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('test_loss', output['loss'], on_epoch = True)
        return output['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class RSampleModule(pl.LightningModule):
    def __init__(self,use_pretrained):
        super().__init__()
        self.unet = unet.UNet(in_channels=3, out_channels=2)

        if(use_pretrained):
            pretrained_state = torch.load('/u/eag-d1/data/Hennepin/model_checkpoints/building_seg_pretrained.pth')

            model_state = self.unet.state_dict()
            pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
            model_state.update(pretrained_state)
            self.unet.load_state_dict(model_state)

        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.unet(x)
        
        means = x[:,0]
        vars = x[:,1]
        vars = self.softplus(vars)

        #stay above 0s
        vars = vars + torch.tensor(1e-16)

        return means, vars # B x H x W ? 

    def value_predictions(self, batch):
        image, masks, values = batch['image'], batch['masks'], batch['values']

        means, vars = self(image)
        # Shape: B X 1 X H X W

        means = torch.flatten(means, start_dim=1)
        vars = torch.flatten(vars, start_dim=1)
        # Shape: B X 1 X HW

        masks = torch.flatten(masks,start_dim = 2)
        masks = torch.swapdims(masks,2,1)
        # Shape: B X HW X 100 'max 100 parcels in a sample"
        
        #Aggregate
        means_sums = torch.matmul(means.unsqueeze(1).float(), masks.float()).squeeze(1)
        #Shape: B X 100

        #We need to ignore the zeroes
        indices = means_sums.nonzero(as_tuple=True)
        means_sums = means_sums[indices]
        values = values[indices]
        #Shape: num_parcels (IN ALL OF BATCH)

        return means_sums, values

    def prob_eval(self, batch, boundary_val):
        image, masks, values = batch['image'], batch['masks'], batch['values']

        means, vars = self(image)
        # Shape: B X 1 X H X W

        means = torch.flatten(means, start_dim=1)
        vars = torch.flatten(vars, start_dim=1)
        # Shape: B X 1 X HW

        masks = torch.flatten(masks,start_dim = 2)
        masks = torch.swapdims(masks,2,1)
        # Shape: B X HW X 100 'max 100 parcels in a sample"

        #Aggregate
        means_sums = torch.matmul(means.unsqueeze(1).float(), masks.float()).squeeze(1)
        vars_sums = torch.matmul(vars.unsqueeze(1).float(), masks.float()).squeeze(1)
        #Shape: B X 100

        #We need to ignore the zeroes
        indices = means_sums.nonzero(as_tuple=True)
        means_sums = means_sums[indices]
        vars_sums = vars_sums[indices]
        values = values[indices]
        #Shape: num_parcels (IN ALL OF BATCH)

        # build the distributions and take the log prob
        gauss = dist.Normal(means_sums, torch.sqrt(vars_sums))
        log_prob = gauss.log_prob(values)
        metric = gauss.cdf(values + boundary_val) - gauss.cdf(values - boundary_val)

        return log_prob, metric

    def shared_step(self, batch):
        image, masks, values = batch['image'], batch['masks'], batch['values']

        means, vars = self(image)
        gauss = dist.Normal(means, torch.sqrt(vars))
        sample = gauss.rsample()
        output = torch.flatten(sample, start_dim=1).unsqueeze(1)

        masks = torch.flatten(masks,start_dim = 2)
        masks = torch.swapdims(masks,2,1)
    
        region_sums_yhat = torch.matmul(output.float(), masks.float()).squeeze(1)

        entropy = -torch.mean(gauss.entropy()) * torch.tensor(cfg.train.lam)
        mean_vars = torch.mean(vars)

        squares = torch.square(values.float()-region_sums_yhat)
        loss = torch.sum(squares, dim=1).mean() + entropy
        
        return {'loss': loss, 'entropy':entropy, 'mean_vars':mean_vars}

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('train_loss', output['loss'], on_epoch = True, batch_size=cfg.train.batch_size)
        self.log('entropy', output['entropy'], on_epoch = True, batch_size=cfg.train.batch_size)
        self.log('mean_vars', output['mean_vars'], on_epoch = True, batch_size=cfg.train.batch_size)
        return output['loss']

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('val_loss', output['loss'], on_epoch = True, batch_size=cfg.train.batch_size)
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

        if(use_pretrained):
            pretrained_state = torch.load('/u/eag-d1/data/Hennepin/model_checkpoints/building_seg_pretrained.pth')

            model_state = self.unet.state_dict()
            pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
            model_state.update(pretrained_state)
            self.unet.load_state_dict(model_state)
        
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.unet(x)
        means = x[:,0]
        vars = x[:,1] 
        vars = self.softplus(vars)
        vars = vars + torch.tensor(1e-16)
        return means, vars

    def value_predictions(self, batch):
        image, masks, values = batch['image'], batch['masks'], batch['values']

        means, vars = self(image)
        # Shape: B X 1 X H X W

        means = torch.flatten(means, start_dim=1)
        vars = torch.flatten(vars, start_dim=1)
        # Shape: B X 1 X HW

        masks = torch.flatten(masks,start_dim = 2)
        masks = torch.swapdims(masks,2,1)
        # Shape: B X HW X 100 'max 100 parcels in a sample"
        
        #Aggregate
        means_sums = torch.matmul(means.unsqueeze(1).float(), masks.float()).squeeze(1)
        variances_sums =  torch.matmul(vars.unsqueeze(1).float(), masks.float()).squeeze(1)
        #Shape: B X 100

        #We need to ignore the zeroes
        indices = means_sums.nonzero(as_tuple=True)
        means_sums = means_sums[indices]
        variances_sums = variances_sums[indices]
        values = values[indices]
        #Shape: num_parcels (IN ALL OF BATCH)

        return means_sums, values

    def prob_eval(self, batch, boundary_val):
        image, masks, values = batch['image'], batch['masks'], batch['values']

        means, vars = self(image)
        # Shape: B X 1 X H X W

        means = torch.flatten(means, start_dim=1)
        vars = torch.flatten(vars, start_dim=1)
        # Shape: B X 1 X HW

        masks = torch.flatten(masks,start_dim = 2)
        masks = torch.swapdims(masks,2,1)
        # Shape: B X HW X 100 'max 100 parcels in a sample"

        #Aggregate
        means_sums = torch.matmul(means.unsqueeze(1).float(), masks.float()).squeeze(1)
        vars_sums = torch.matmul(vars.unsqueeze(1).float(), masks.float()).squeeze(1)
        #Shape: B X 100

        #We need to ignore the zeroes
        indices = means_sums.nonzero(as_tuple=True)
        means_sums = means_sums[indices]
        vars_sums = vars_sums[indices]
        values = values[indices]
        #Shape: num_parcels (IN ALL OF BATCH)

        # build the distributions and take the log prob
        gauss = dist.Normal(means_sums, torch.sqrt(vars_sums))
        log_prob = gauss.log_prob(values)
        metric = gauss.cdf(values + boundary_val) - gauss.cdf(values - boundary_val)

        return log_prob, metric

    def shared_step(self, batch): 
        image, masks, values = batch['image'], batch['masks'], batch['values']

        means, vars = self(image)
        mean_vars = torch.mean(vars)
        # Shape: B X 1 X H X W

        means = torch.flatten(means, start_dim=1)
        vars = torch.flatten(vars, start_dim=1)
        # Shape: B X 1 X HW

        masks = torch.flatten(masks,start_dim = 2)
        masks = torch.swapdims(masks,2,1)
        # Shape: B X HW X 100 'max 100 parcels in a sample"

        #Aggregate
        means_sums = torch.matmul(means.unsqueeze(1).float(), masks.float()).squeeze(1)
        variances_sums =  torch.matmul(vars.unsqueeze(1).float(), masks.float()).squeeze(1)
        #Shape: B X 100

        squares = torch.square(values.float()-means_sums)

        #We need to ignore the zeroes
        indices = means_sums.nonzero(as_tuple=True)
        means_sums = means_sums[indices]
        variances_sums = variances_sums[indices]
        values = values[indices]
        #Shape: num_parcels (IN ALL OF BATCH)

        stds_sums = torch.sqrt(variances_sums)
        gauss = dist.Normal(means_sums, stds_sums)
        
        # Loss1 = Batch_mean( Average of log_probabities of the true values for parcels in the image divided by num_parcels )
        # Loss2 = Batch_mean( sum of square errors of predicted means and true values for parcels in the image. )
        # lamda = 1e-3 
        
        loss =  -torch.sum(gauss.log_prob(values))/len(indices) #+ torch.sum(squares, dim=1).mean() * torch.tensor(1e-3)

        return {'loss': loss, 'mean_vars': mean_vars}

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('train_loss', output['loss'], on_epoch = True, batch_size=cfg.train.batch_size)
        self.log('mean_vars', output['mean_vars'], on_epoch = True, batch_size=cfg.train.batch_size)
        return output['loss'] 

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('val_loss', output['loss'], on_epoch = True, batch_size=cfg.train.batch_size)
        return output['loss']

    def test_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('test_loss', output['loss'], on_epoch = True)
        return output['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class LOGRSampleModule(pl.LightningModule):
    def __init__(self,use_pretrained):
        super().__init__()
        self.unet = unet.UNet(in_channels=3, out_channels=2)

        if(use_pretrained):
            pretrained_state = torch.load('/u/eag-d1/data/Hennepin/model_checkpoints/building_seg_pretrained.pth')

            model_state = self.unet.state_dict()
            pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
            model_state.update(pretrained_state)
            self.unet.load_state_dict(model_state)

        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.unet(x)
        
        means = x[:,0]
        vars = x[:,1]
        vars = self.softplus(vars)

        #stay above 0s
        vars = vars + torch.tensor(1e-16)

        return means, vars # B x H x W ? 

    def value_predictions(self, batch):
        image, masks, values = batch['image'], batch['masks'], batch['values']

        means, vars = self(image)
        # Shape: B X 1 X H X W

        means = torch.flatten(means, start_dim=1)
        vars = torch.flatten(vars, start_dim=1)
        # Shape: B X 1 X HW

        masks = torch.flatten(masks,start_dim = 2)
        masks = torch.swapdims(masks,2,1)
        # Shape: B X HW X 100 'max 100 parcels in a sample"
        
        #Aggregate
        means_sums = torch.matmul(means.unsqueeze(1).float(), masks.float()).squeeze(1)
        #Shape: B X 100

        #We need to ignore the zeroes
        indices = means_sums.nonzero(as_tuple=True)
        means_sums = means_sums[indices]
        values = values[indices]
        #Shape: num_parcels (IN ALL OF BATCH)

        return means_sums, values

    def prob_eval(self, batch, boundary_val):
        image, masks, values = batch['image'], batch['masks'], batch['values']

        means, vars = self(image)
        # Shape: B X 1 X H X W

        means = torch.flatten(means, start_dim=1)
        vars = torch.flatten(vars, start_dim=1)
        # Shape: B X 1 X HW

        masks = torch.flatten(masks,start_dim = 2)
        masks = torch.swapdims(masks,2,1)
        # Shape: B X HW X 100 'max 100 parcels in a sample"

        #Aggregate
        means_sums = torch.matmul(means.unsqueeze(1).float(), masks.float()).squeeze(1)
        vars_sums = torch.matmul(vars.unsqueeze(1).float(), masks.float()).squeeze(1)
        #Shape: B X 100

        #We need to ignore the zeroes
        indices = means_sums.nonzero(as_tuple=True)
        means_sums = means_sums[indices]
        vars_sums = vars_sums[indices]
        values = values[indices]
        #Shape: num_parcels (IN ALL OF BATCH)

        # build the distributions and take the log prob
        gauss = dist.Normal(means_sums, torch.sqrt(vars_sums))
        log_prob = gauss.log_prob(values)
        metric = gauss.cdf(values + boundary_val) - gauss.cdf(values - boundary_val)

        return log_prob, metric

    def shared_step(self, batch):
        image, masks, values = batch['image'], batch['masks'], batch['values']

        means, vars = self(image)
        gauss = dist.Normal(means, torch.sqrt(vars))

        # Take 100 samples
        samples = []
        for i in range(0,100):
            sample = gauss.rsample()
            output = torch.flatten(sample, start_dim=1).unsqueeze(1)
            samples.append(output)

        masks = torch.flatten(masks,start_dim = 2)
        masks = torch.swapdims(masks,2,1)

        #Aggregate
        region_sums = []
        for i in range(0,100):
            region_sums_yhat = torch.matmul(output.float(), masks.float()).squeeze(1)
            region_sums.append(region_sums_yhat)

        #Compute statistsics of samples
        region_sums = torch.stack(region_sums)
        mean = torch.mean(region_sums, dim = 0)
        std = torch.std(region_sums, dim = 0)

        #print(values.shape)
        #print(std.shape)
        #print(mean.shape)

        #We need to ignore the zeroes
        indices = mean.nonzero(as_tuple=True)
        mean = mean[indices]
        std = std[indices]
        values = values[indices]
        #Shape: num_parcels (IN ALL OF BATCH)

        #Avoid std = 0
        std = std + torch.tensor(1)
        gauss = dist.Normal(mean,std)
        #print(std)
        
        loss =  -torch.sum(gauss.log_prob(values))/len(indices)
        #loss = 0
        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('train_loss', output['loss'], on_epoch = True, batch_size=cfg.train.batch_size)
        #self.log('mean_vars', output['mean_vars'], on_epoch = True, batch_size=cfg.train.batch_size)
        return output['loss']

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('val_loss', output['loss'], on_epoch = True, batch_size=cfg.train.batch_size)
        return output['loss']

    def test_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('test_loss', output['loss'], on_epoch = True)
        return output['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer