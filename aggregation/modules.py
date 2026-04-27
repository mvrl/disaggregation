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

        self.agg = util.regionAgg_layer()

    def forward(self, x):
        x = self.unet(x)
        x = torch.flatten(x, start_dim=1)
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
        image, parcel_masks, parcel_values = batch

        output = self(image)
        
        #take cnn output and parcel masks(Aggregation Matrix M)
        estimated_values = self.agg(output, parcel_masks)
        loss = util.MSE(estimated_values, parcel_values)
        
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
        self.agg = util.regionAgg_layer()

    def get_valOut(self, x):
        means, vars = self(x)
        
        return means.unsqueeze(1), vars.unsqueeze(1)

    def forward(self, x):
        x = self.unet(x)
        #x = self.softplus(x)
        
        means = x[:,0]
        vars = x[:,1]
        vars = self.softplus(vars)

        #stay above 0
        vars = vars + 1e-16

        return means, vars

    def pred_Out(self, x, masks):
        means, vars = self(x)
        output = torch.flatten(means, start_dim=1)
        estimated_values = self.agg(output, masks)
        return estimated_values

    def log_out(self, x, masks, targets):
        means, vars = self(x)
        means = torch.flatten(means, start_dim=1)
        vars = torch.flatten(vars, start_dim=1)
        means= self.agg(means, masks)
        vars = self.agg(vars, masks)
       
        losses=[]
        for mean,var,target in zip(means,vars, targets):
            std = torch.sqrt(var)
            gauss = dist.Normal(mean, std)
            losses.append(torch.exp(gauss.log_prob(target.cuda())))
        return losses

    def shared_step(self, batch):
        image, parcel_masks, parcel_values = batch

        means, vars = self(image)
        gauss = dist.Normal(means, torch.sqrt(vars))
        sample = gauss.rsample()
        output = torch.flatten(sample, start_dim=1)

        entropy = -torch.mean(gauss.entropy()) * torch.tensor(cfg.train.lam)
        mean_vars = torch.mean(vars)
        
        #take cnn output and parcel masks(Aggregation Matrix M)
        estimated_values = self.agg(output, parcel_masks)
        loss = util.MSE(estimated_values, parcel_values) + entropy

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
        self.agg = util.regionAgg_layer()

    def forward(self, x):
        x = self.unet(x)
        means = x[:,0]
        vars = x[:,1] 
        vars = self.softplus(vars)
        vars = vars + 1e-16
        means = torch.flatten(means, start_dim=1)
        vars = torch.flatten(vars, start_dim=1)
        # Shape: B X 1 X HW

        masks = torch.flatten(masks,start_dim = 2)
        masks = torch.swapdims(masks,2,1)
        # Shape: B X HW X 100 'max 100 parcels in a sample"
        
        #Aggregate
        means_sums = torch.matmul(means.unsqueeze(1).float(), masks.float()).squeeze(1)
        stds = vars**.5
        variances_sums =  torch.matmul(stds.unsqueeze(1).float(), masks.float()).squeeze(1)**2

        #We need to ignore the zeroes
        indices = means_sums.nonzero(as_tuple=True)
        means_sums = means_sums[indices]
        variances_sums = variances_sums[indices]
        values = values[indices]
        #Shape: num_parcels (IN ALL OF BATCH)

        return means_sums, values

    def prob_eval(self, batch, boundary_val, boundary_val2):
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

        #Off by a factor of N?
        vars_sums = vars_sums * torch.count_nonzero(masks, dim=1)

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
        metric2 = gauss.cdf(values + boundary_val2) - gauss.cdf(values - boundary_val2)
        return log_prob, metric, metric2, torch.sqrt(vars_sums)

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

        #Off by a factor of N?
        variances_sums = variances_sums * torch.count_nonzero(masks, dim=1)
        

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
    def __init__(self,use_pretrained, num_samples):
        super().__init__()
        self.unet = unet.UNet(in_channels=3, out_channels=2)

        if(use_pretrained):
            pretrained_state = torch.load('/u/eag-d1/data/Hennepin/model_checkpoints/building_seg_pretrained.pth')

            model_state = self.unet.state_dict()
            pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
            model_state.update(pretrained_state)
            self.unet.load_state_dict(model_state)

        self.softplus = nn.Softplus()

        self.one_sample_std = torch.nn.Parameter(torch.tensor(5000.0))
        self.one_sample_std.requires_grad = True
        

        self.num_samples = num_samples

    def get_valOut(self, x):
        x = self.unet(x)
        means = x[:,0]
        vars = x[:,1] 
        vars = self.softplus(vars)
        return means.unsqueeze(1), vars.unsqueeze(1)

    def pred_Out(self, x, masks):
         means, vars = self(x)
         estimated_values = self.agg(means, masks)
         return estimated_values

    def log_out(self, x, masks, targets):
        means, vars = self(x)
        means= self.agg(means, masks)
        vars = self.agg(vars, masks)

        losses=[]
        for mean,var,target in zip(means,vars, targets):
            std = torch.sqrt(var)
            gauss = dist.Normal(mean, std)
            losses.append(torch.exp(gauss.log_prob(target.cuda())))
        return losses


    def shared_step(self, batch): 
        image, parcel_masks, parcel_values = batch

        means, vars = self(image)
        
        #take cnn output and parcel masks(Aggregation Matrix M)
        means= self.agg(means, parcel_masks)
        vars = self.agg(vars, parcel_masks)

        loss = util.gaussLoss(means, vars, parcel_values)
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