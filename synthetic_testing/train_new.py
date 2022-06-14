import pytorch_lightning as pl
import torch
import torch.distributions as dist
import torch.nn as nn
from dataset import Eurosat
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from unet import UNet
from argparse import ArgumentParser, Namespace
from pytorch_lightning import seed_everything
import test_new 


def MSE(outputs, targets):
    
    losses = (outputs - targets)**2
    losses = torch.mean(losses)
    
    return losses

def gaussLoss_train(mean, std, target):
    gauss = dist.Normal(mean, std)
    loss = gauss.log_prob(target)
    loss = -torch.mean(loss)
    return loss


class RegionAggregator(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        self.unet = UNet(3, 2)
        self.softplus = nn.Softplus()
        self.avg_pool = nn.AvgPool2d((hparams.kernel_size, hparams.kernel_size), stride=hparams.kernel_size)
        self.flatten = nn.Flatten()

        self.trainset = Eurosat(mode='train')
        self.valset = Eurosat(mode='validation')
        self.save_hyperparameters(hparams)
    def forward(self, x):
        x = self.unet(x)

        mean = x[:, 0]
        std = x[:, 1]
        std = self.softplus(std) + 1e-16
        
        return mean, std

    def shared_step(self, batch):
        images = batch['image']
        labels = batch['label']

        # log_prob_orig = -test_new.gaussLoss_test(self.pred_Out(images)[0], 
        #                                 self.pred_Out(images)[1],
        #                                 self.flatten(labels))
        
        labels = self.avg_pool(labels)
        labels = self.flatten(labels)
        mu, std = self(images)
        
        
        mean_std = torch.mean(std)
        # log_prob = -test_new.gaussLoss_test(mu, std, labels)
        

        loss = gaussLoss_train(mu, std, labels)
        
        return {'loss': loss, 'mean_std': mean_std, 
                # 'log_prob': log_prob, 'log_prob_orig': log_prob_orig
                }


    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch)

        self.log('train_loss', output['loss'])
        self.log('std', output['mean_std'])
        # self.log('log_prob', output['log_prob'])
        # self.log('log_prob_orig', output['log_prob_orig'])
        return output['loss']

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('val_loss', output['loss'])
        return output['loss']

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset,
                                           batch_size=self.hparams.batch_size,
                                           num_workers=self.hparams.workers,
                                           shuffle=True,
                                           pin_memory=False
                                           )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valset,
                                           batch_size=self.hparams.batch_size,
                                           num_workers=self.hparams.workers,
                                           pin_memory=False
                                           )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        return {'optimizer': optimizer}

    def gauss_fit(self):
        
        import scipy
        from scipy.optimize import curve_fit
        from scipy.stats import norm

        data_y = []
        data_x = []
        
        for i in range(self.trainset.__len__()):
            data_y.append(self.trainset.__getitem__(i)['label'].unsqueeze(0).numpy())
        mean, std= scipy.stats.norm.fit( data_y)
        gauss = torch.distributions.Normal(mean,std)
        return mean, std, gauss

class SamplingRegionAggregator(RegionAggregator):

    def __init__(self, k: int, hparams):
        super().__init__(hparams)
        self.k = k

    def gaussian(self, mu, std):
        gauss_dist = dist.Normal(mu, std)
        return gauss_dist.rsample((self.k,))

    def compute_per_region(self, sample):
        est = self.avg_pool(sample)
        est = torch.flatten(est, 2, 3)
        mean_d = torch.mean(est, 0)
        var_d = torch.var(est, 0) * self.hparams.kernel_size ** 2
        std_d = torch.sqrt(var_d)
        return mean_d, std_d

    def forward(self, x):
        mu, std = super().forward(x)
        sample = self.gaussian(mu, std)
        mean, std = self.compute_per_region(sample)
        return mean, std

    def pred_Out(self, x):
        means, std = super().forward(x)
        means = torch.flatten (means, 1,2)
        std = torch.flatten (std, 1,2)
        return means, std


class AnalyticalRegionAggregator(RegionAggregator):

    def __init__(self, hparams):
        super().__init__(hparams)

    def forward(self, x):
        mu, std = super().forward(x)
        means = self.avg_pool(mu)
        var = self.avg_pool(std**2) / (self.hparams.kernel_size**2)

        means = torch.flatten(means, 1, 2)
        std = torch.flatten(torch.sqrt(var), 1, 2)
        return means, std

    def pred_Out(self, x):
        means, std = super().forward(x)
        means = torch.flatten (means, 1,2)
        std = torch.flatten (std, 1,2)
        return means, std

class Uniform_model(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        self.unet = UNet(3, 2)
        self.softplus = nn.Softplus()
        self.avg_pool = nn.AvgPool2d((hparams.kernel_size, hparams.kernel_size), stride=hparams.kernel_size)
        self.flatten = nn.Flatten()
        self.trainset = Eurosat(mode='train')
        self.valset = Eurosat(mode='validation')
        self.save_hyperparameters(hparams)

    def forward(self, x):
        x = self.unet(x)

        mean = x[:, 0]  
        std = x[:, 1] 
        std = self.softplus(std) + 1e-16

        return mean, std

    def shared_step(self, batch):
        images = batch['image']
        labels = batch['label']

       # log_prob_orig = -gaussLoss_test(self.pred_Out(images)[0],
       #                                 self.pred_Out(images)[1],
       #                                 self.flatten(labels))

        labels = self.avg_pool(labels)
        labels = labels
        labels = torch.nn.functional.upsample_nearest(labels.unsqueeze(1),scale_factor=self.hparams.kernel_size)
        labels = labels.squeeze(1)

        mu, std = self(images)

        mean_std = torch.mean(std)

        loss = gaussLoss_train(mu, std, labels)

        return {'loss': loss, 'mean_std': mean_std}


    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch)

        self.log('train_loss', output['loss'])
        self.log('std', output['mean_std'])
        return output['loss']

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('val_loss', output['loss'])
        return output['loss']

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset,
                                           batch_size=self.hparams.batch_size,
                                           num_workers=self.hparams.workers,
                                           shuffle=True,
                                           pin_memory=False
                                           )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valset,
                                           batch_size=self.hparams.batch_size,
                                           num_workers=self.hparams.workers,
                                           pin_memory=False
                                           )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        return {'optimizer': optimizer}

    
    def pred_Out(self, x):
        means, std = self(x)
        means = torch.flatten (means, 1,2)
        std = torch.flatten (std, 1,2)
        return means, std



class deterministic_model(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        self.unet = UNet(3, 1)
        self.softplus = nn.Softplus()
        self.avg_pool = nn.AvgPool2d((hparams.kernel_size, hparams.kernel_size), stride=hparams.kernel_size)
        self.flatten = nn.Flatten()
        self.trainset = Eurosat(mode='train')
        self.valset = Eurosat(mode='validation')
        self.save_hyperparameters(hparams)

    def forward(self, x):
        x = self.unet(x)

        return x

    def shared_step(self, batch):
        images = batch['image']
        labels = batch['label']

       # log_prob_orig = -gaussLoss_test(self.pred_Out(images)[0],
       #                                 self.pred_Out(images)[1],
       #                                 self.flatten(labels))

        labels = self.avg_pool(labels)
        labels = self.flatten(labels)

        net_out  = self(images)
        net_out = self.avg_pool(net_out)
        net_out = self.flatten(net_out)

        loss = MSE(net_out,labels)

        return {'loss': loss}


    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch)

        self.log('train_loss', output['loss'])
        return output['loss']

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        self.log('val_loss', output['loss'])
        return output['loss']

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset,
                                           batch_size=self.hparams.batch_size,
                                           num_workers=self.hparams.workers,
                                           shuffle=True,
                                           pin_memory=False
                                           )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valset,
                                           batch_size=self.hparams.batch_size,
                                           num_workers=self.hparams.workers,
                                           pin_memory=False
                                           )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        return {'optimizer': optimizer}

    
    def pred_Out(self, x):
        net_out = self(x)
        net_out = self.flatten (net_out)
        return net_out


def main(args):
    if type(args) == dict:
        args = Namespace(**args)
    method = args.method
    seed_everything(args.seed, workers=True)

    log_dir = '{}/{}/{}/{}/{}'.format(
        'average',
        args.save_dir,
        args.method,
        args.kernel_size,
        args.samples
    )

    logger = TensorBoardLogger(log_dir)

    if args.method == 'analytical':
        model = AnalyticalRegionAggregator(args)
    elif args.method == 'rsample':
        model = SamplingRegionAggregator(args.samples, args)
    elif args.method == 'uniform':
        model = Uniform_model(args)
    elif args.method == 'det':
        model = deterministic_model(args)
    
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1)

    early_stopping = EarlyStopping('val_loss', patience=args.patience)

    trainer = pl.Trainer.from_argparse_args(args,
                                            max_epochs=args.max_epochs,
                                            logger=logger,
                                            callbacks=[checkpoint_callback, early_stopping],
                                            deterministic=True
                                            )
    trainer.fit(model)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=150)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=.01)
    parser.add_argument('--save_dir', default='logtest')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=16)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=80)
    parser.add_argument('--patience', type=int, default=100)

    parser.add_argument('--method', type=str, default='analytical')
    parser.add_argument('--lambdaa', type=float, default=0.)
    args = parser.parse_args()
    main(args)

