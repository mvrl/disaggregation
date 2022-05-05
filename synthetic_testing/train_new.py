import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torchmetrics

import numpy as np
from argparse import ArgumentParser, Namespace
from matplotlib import pyplot as plt

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from dataset import Eurosat

from unet import UNet

from shutil import copyfile
import os



def gaussLoss_train(mean, std, target):
    
    gauss = dist.Normal(mean, std)
    loss = gauss.log_prob(target)
    loss = -(torch.sum(loss, 1))
    loss = torch.mean(loss)
    return loss


def gaussLoss_test(mean, std, target):
    
    gauss = dist.Normal(mean, std)
    loss = gauss.log_prob(target)
    loss = -(torch.mean(loss, 1))
    loss = torch.mean(loss)
    return loss


class RegionAggregator(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
       
        if type(hparams)==dict:
            hparams = Namespace(**hparams)
        
        self.unet = UNet(3, 2) 
        self.softplus = nn.Softplus()
        self.avg_pool = nn.AvgPool2d((hparams.kernel_size,hparams.kernel_size), stride=hparams.kernel_size)
        self.flatten =  nn.Flatten()
                                      
        self.trainset = Eurosat(mode='train')
        self.valset = Eurosat(mode='validation') 
        self.save_hyperparameters(hparams)

    def forward(self, x):
        
        x = self.unet(x)

        mean = x[:, 0]
        var = x[:, 1]
        var = self.softplus(var) + 1e-16

        return mean, torch.sqrt( var )

    def shared_step(self, batch):
        
        images = batch['image']
        labels = batch['label']

        labels = self.avg_pool(labels)
        labels = self.flatten(labels)
        mu, var = self (images)
 
        loss = gaussLoss_train(mu, var, labels) 

        return {'loss': loss}
    
    def pred_Out(self, x):
       
        means, var  = self(x)
        means = torch.flatten (means, 1,2)
        var = torch.flatten (var, 1,2)
        return means, var

    def agg_labels(self, labels):
        
        labels = self.avg_pool(labels)# * self.hparams.kernel_size**2
        agg_labels = self.flatten(labels)
        
        return agg_labels

    def training_step(self, batch, batch_idx):
        
        output = self.shared_step(batch)
        
        self.log('train_loss', output['loss'],  
                on_epoch = True, batch_size=self.hparams.batch_size)
        
        return output['loss']

    def validation_step(self, batch, batch_idx):
        
        output = self.shared_step(batch)
        self.log('val_loss', output['loss'])
        return output['loss']


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset, 
                batch_size = self.hparams.batch_size,
                num_workers = self.hparams.workers,
                shuffle=True,
                pin_memory=False
                )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valset, 
                batch_size = 1,
                num_workers = self.hparams.workers,
                pin_memory=False
                )
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        
        return {'optimizer': optimizer}


class SamplingRegionAggregator(RegionAggregator):

    def __init__(self, k:int, hparams):
        super().__init__( hparams )
        self.k = k
       
     
    def gaussian(self,mu, std):
            
        sample = torch.zeros(self.k,mu.shape[0],mu.shape[1], mu.shape[2]).to('cuda')

        gauss_dist = dist.Normal(mu, std)

        for i in range (self.k):
            sample[i] = gauss_dist.rsample()
        return sample

    def compute_per_region(self,sample):
        
        est = self.avg_pool(sample) 
        est = torch.flatten (est, 2,3)
        mean_d = torch.mean (est, 0)
        std_d = torch.std (est, 0)
        return mean_d, std_d
    
    def forward(self, x):
        mu, std = super().forward( x )
        sample = self.gaussian(mu, std)
        mean, std = self.compute_per_region(sample)
        return mean, std


class AnalyticalRegionAggregator(RegionAggregator):

    def __init__(self, hparams):
        super().__init__( hparams )

    def forward(self, x):
        mu, std = super().forward( x )
        means = self.avg_pool(mu)
        std = self.avg_pool( std )# * self.hparam

        means = torch.flatten (means, 1,2)
        std = torch.flatten(std, 1,2)
        return means, std


def main(args):

    if type(args)==dict:
            args = Namespace(**args)
    method = args.method
    log_dir = '{}/{}/{}/{}'.format(
        args.save_dir,
        args.method,
        args.kernel_size,
        args.samples
        )

    logger = TensorBoardLogger(log_dir)
    
    if args.method == 'analytical':
        model = AnalyticalRegionAggregator(args)
    elif args.method == 'rsample':
        model = SamplingRegionAggregator(10,args)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1)
    
    early_stopping = EarlyStopping('val_loss', patience=args.patience)

    best_path = checkpoint_callback.best_model_path
    trainer = pl.Trainer.from_argparse_args(args,
                                            max_epochs = args.max_epochs,
                                            logger=logger, 
                                            callbacks=[checkpoint_callback, early_stopping],
                                            )
    trainer.fit(model)    
    
if __name__ == '__main__':
    from argparse import ArgumentParser, Namespace 
    
    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=.01)
    parser.add_argument('--save_dir', default='new_logs')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=16)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--patience', type=int, default=100)
                         
    parser.add_argument('--method', type=str, default='analytical')
    parser.add_argument('--lambdaa', type=float, default=0.)
    args = parser.parse_args()
    main(args)
