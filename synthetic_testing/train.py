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
    
    losses = (outputs - targets)**2
    losses = (torch.sum(losses, 2).sum(1))
    losses = torch.mean(losses)
    
    return losses

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
        var = self.softplus(var)+1e-16


        return mean, var


    def pred_Out(self, x):
       
        means, var  = self(x)
        means = torch.flatten (means, 1,2)
        var = torch.flatten (var, 1,2)
        return means, var
        
    

    def agg_labels(self, labels):
        
        labels = self.avg_pool(labels)# * self.hparams.kernel_size**2
        agg_labels = self.flatten(labels)
        
        return agg_labels


    def shared_step(self, batch):
        
        images = batch['image']
        labels = batch['label']
        
        means, var = self (images)
        mean_var = torch.mean(var)  
        mse = (torch.square (means-labels)).mean()
        
        if (self.hparams.method == "analytical" or self.hparams.method == "rsample"):
            labels = self.avg_pool(labels)# * self.hparams.kernel_size**2
        
        if (self.hparams.method == "interpolate"):
            agg_labels = self.avg_pool(labels)# * self.hparams.kernel_size**2
            labels = torch.nn.functional.interpolate(agg_labels.unsqueeze(1), size=(64,64), mode='bicubic')
            labels = labels.squeeze(1) 
        
        if (self.hparams.method == "full_res"):
            labels = labels

       # print (mae, "mae")
       # print (means, "means")
       # print (var, "var")
       # print (labels, "labels")
        if (self.hparams.method == "analytical"):
            means = self.avg_pool(means)# * self.hparams.kernel_size**2
            var = self.avg_pool(var)# * self.hparams.kernel_size**2
            
            means = torch.flatten (means, 1,2)
            std = torch.sqrt (torch.flatten (var, 1,2))
            labels = self.flatten(labels)

          #  means = self.flatten(agg_mean)
          #  var = self.flatten(agg_var)
             
        
        if ( self.hparams.method == "interpolate" or self.hparams.method =="full_res" or self.hparams.method == "rsample"):
            
            k = 500
            sample = torch.zeros(k,means.shape[0],means.shape[1], means.shape[2]).to('cuda')
            
            gauss_dist = dist.Normal(means, torch.sqrt(var))
            
            for i in range (k):
                sample[i] = gauss_dist.rsample()
            
       #     print ("sample shape",sample.shape)
            
            entropy =  -gauss_dist.entropy().mean()
             
        if (self.hparams.method == "rsample"):
            
            est = self.avg_pool(sample)
            est = torch.flatten (est, 2,3)
    #        print (est.shape)
            labels = self.flatten(labels)
    #        print (labels.shape) 
            mean_d = torch.mean (est, 0)
            std_d = torch.std (est, 0)
    #        print (mean_d.shape, std_d.shape, "sh") 
         #   print (mean_d.shape, std_d.shape, labels.shape)
            loss = gaussLoss_train(mean_d, std_d, labels) + self.hparams.lambdaa*entropy
           # loss = MSE(est, labels) + self.hparams.lambdaa*entropy

        elif (self.hparams.method == "interpolate" or self.hparams.method =="full_res"):
           # print (sample.shape, labels.shape)
            est = torch.flatten (sample, 2,3)
           # print (est.shape, "est")
            labels = self.flatten(labels)
           # print (labels.shape, "label")
            mean_d = torch.mean (est, 0)
            std_d = torch.std (est, 0)

           # print (mean_d.shape, std_d.shape, labels.shape, "all")
            loss = gaussLoss_train(mean_d, std_d, labels) + self.hparams.lambdaa*entropy

           # loss = MSE(sample, labels) +1e3* kl(sample, dist.Normal(torch.zeros(sample.shape[0], sample.shape[1], sample.shape[2]).to("cuda")
           #     ,torch.ones(sample.shape[0], sample.shape[1], sample.shape[2]).to("cuda")).rsample()) 
        else :
         #   print (means,var, "meanvar")
            loss = gaussLoss_train(means, std, labels)
        
        if (self.hparams.method =="analytical"):
            entropy = 0

        return {'loss': loss, 'entropy': entropy,
                "mean_var": mean_var,
                'mse': mse}
    
    def training_step(self, batch, batch_idx):
        
        output = self.shared_step(batch)
        self.log('train_loss', output['loss'],  
                on_epoch = True, batch_size=self.hparams.batch_size)
        self.log('entropy', output['entropy'],  
                on_epoch = True, batch_size=self.hparams.batch_size)
        self.log('var', output['mean_var'],
                on_epoch = True, batch_size=self.hparams.batch_size)
        self.log('mse', output['mse'],
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
    
    def gauss_fit(self):
        
        import scipy
        from scipy.optimize import curve_fit
        from scipy.stats import norm

        data_y = []
        data_x = []
        
        for i in range(self.trainset.__len__()):
            data_y.append(self.avg_pool(self.trainset.__getitem__(i)['label'].unsqueeze(0)).numpy())
            data_x.append(self.trainset.__getitem__(i)['image'].unsqueeze(0).numpy())
        mean, std= scipy.stats.norm.fit( data_y)
        print (data_x[0], data_y[0], data_y[0].max(), data_y[0].min(), data_y[0].std(),mean, std)
        return mean, std

def main(args):
                             
    if type(args)==dict:
            args = Namespace(**args)
    method = args.method
    log_dir = '{}/{}/{}'.format(
        args.save_dir,
        args.method,
        args.kernel_size
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
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=.01)
    parser.add_argument('--save_dir', default='new_logs')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=16)
    parser.add_argument('--patience', type=int, default=100)
                         
    parser.add_argument('--method', type=str, default='analytical')
    parser.add_argument('--lambdaa', type=float, default=0.)
    args = parser.parse_args()
    main(args)
