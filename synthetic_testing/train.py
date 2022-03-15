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



def MSE(outputs, targets):
    losses = []
    for output,target in zip(outputs,targets):
        losses.append( torch.sum((output - target)**2) )
    loss = torch.stack(losses, dim=0).mean()
    return loss


def gaussLoss(means, vari, targets):
    losses = []
    for mean,var,target in zip(means,vari,targets):
        std = torch.sqrt(var)
        gauss = dist.Normal(mean, std)
        losses.append(-torch.sum(gauss.log_prob(target)))
    loss = torch.stack(losses, dim=0).mean()
    return loss

class regionize_gauss(pl.LightningModule):
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

        var = self.softplus(var)

        if (self.hparams.method == "analytical"):
            agg_mean = self.avg_pool(mean)# * self.hparams.kernel_size**2
            agg_var = self.avg_pool(var)# * self.hparams.kernel_size**2

            agg_mean = self.flatten(agg_mean)
            agg_var = self.flatten(agg_var)
        
            return  agg_mean, agg_var

        else:   
            return mean, var

    
    def pred_Out(self, x):
       
        means, _  = self(x)

        if (self.hparams.method == "rsample"):
            means = self.avg_pool(mean)
            means = self.flatten(agg_mean)

        return means
        
    

    def log_out(self, x, targets):
        
        means, vari = self(x)

        losses=[]
        for mean,var,target in zip(means,vari, targets):
            std = torch.sqrt(var)
            gauss = dist.Normal(mean, std)
            losses.append(torch.exp(gauss.log_prob(target)))
        return losses
    
    def agg_labels(self, labels):
        
        labels = self.avg_pool(labels)# * self.hparams.kernel_size**2
        agg_labels = self.flatten(labels)
        
        return agg_labels


    def shared_step(self, batch):
        
        images = batch['image']
        labels = batch['label']
        
        
        if (self.hparams.method == "analytical" or self.hparams.method == "rsample"):
            agg_labels = self.avg_pool(labels)# * self.hparams.kernel_size**2
            labels = self.flatten(agg_labels)
        
        
        if (self.hparams.method == "interpolate"):
            agg_labels = self.avg_pool(labels)# * self.hparams.kernel_size**2
            labels = torch.nn.functional.interpolate(agg_labels.unsqueeze(1), size=(64,64), mode='bicubic')

        
        if (self.hparams.method == "full res"):
            labels = labels

        means, var = self (images)

        if (self.hparams.method == "rsample"):
            
            gauss_dist = dist.Normal(means, torch.sqrt(var))
            sample = gauss_dist.rsample()

            entropy = -torch.mean(gauss_dist.entropy())
            mean_var = torch.mean(var)

            est = self.avg_pool(sample)
            est = self.flatten(est)
         
        if (self.hparams.method == "rsample"):
            loss = MSE(est, labels) + entropy
        else:
            loss = gaussLoss(means, var, labels)
                
        return {'loss': loss}
    
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
 
def main(args):
                             
    if type(args)==dict:
            args = Namespace(**args)
    
    method = args.method
    log_dir = '{}/{}'.format(
        args.save_dir,
        args.method,
        )

    logger = TensorBoardLogger(log_dir)
    
    
    model = regionize_gauss(hparams=args)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=3, save_last=True)
    
    early_stopping = EarlyStopping('val_loss', patience=args.patience)

    best_path = checkpoint_callback.best_model_path
    
    trainer = pl.Trainer.from_argparse_args(args,
                                            max_epochs = args.max_epochs,
                                            logger=logger, 
                                            callbacks=[checkpoint_callback, early_stopping],
                                            )
    trainer.fit(model)    
    print ("best path is saved in ", best_path)
    
if __name__ == '__main__':
    from argparse import ArgumentParser, Namespace 
    
    parser = ArgumentParser()
    
    parser.add_argument('--max_epochs', type=int, default=150)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=.0001)
    parser.add_argument('--save_dir', default='logs')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=16)
    parser.add_argument('--patience', type=int, default=100)
                         
    parser.add_argument('--method', type=str, default='analytical')
    args = parser.parse_args() 
    main(args)
