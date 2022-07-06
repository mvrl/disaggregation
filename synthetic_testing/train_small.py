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



def gaussLoss(mean, std, target):
    gauss = dist.Normal(mean, std)
    loss = gauss.log_prob(target) 
    loss = -torch.mean(loss)
    return loss


class model(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        self.unet = UNet(3, 2)
        self.softplus = nn.Softplus()
        self.avg_pool = nn.AvgPool2d((hparams.kernel_size, hparams.kernel_size), stride=hparams.kernel_size)
        self.sum_pool = nn.AvgPool2d((hparams.kernel_size, hparams.kernel_size), stride=hparams.kernel_size, divisor_override = 1 )
        self.flatten = nn.Flatten()

        self.trainset = Eurosat(mode='train')
        self.valset = Eurosat(mode='validation')
        self.save_hyperparameters(hparams)

    def forward(self, x):
        x = self.unet(x)

        mean = x[:, 0]
        std = x[:, 1]
        std = self.softplus(std) + 1e-8
 
        return mean, std

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch)

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


class AnalyticalRegionAggregator(model):

    def forward(self, x):
        mu, std = super().forward(x)
        
        gauss_dist = dist.Normal(mu, std)

        means = self.sum_pool(mu)
        var = self.sum_pool(std**2)

        means = torch.flatten(means, 1, 2)
        std = torch.flatten(torch.sqrt(var), 1, 2)
        return means, std

    def shared_step(self, batch):
        images = batch['image']
        labels = batch['label']

        labels = self.sum_pool(labels)
        labels = self.flatten(labels)
        
        mu, std = self(images)
        
        loss = gaussLoss(mu, std, labels)
        
        return {'loss': loss}

    def pred_Out(self, x):
        means, std = super().forward(x)
        means = torch.flatten (means, 1,2)
        std = torch.flatten (std, 1,2)
        return means, std


class Uniform_model(model):

    def shared_step(self, batch):
        images = batch['image']
        labels = batch['label']

        labels = self.avg_pool(labels)
        labels = torch.nn.functional.upsample_nearest(labels.unsqueeze(1),scale_factor=self.hparams.kernel_size)
        labels = labels.squeeze(1)
        
        mu, std = self(images)

        loss = gaussLoss(mu, std, labels)

        return {'loss': loss}

    def pred_Out(self, x):
        means, std = self(x)
        means = torch.flatten (means, 1,2)
        std = torch.flatten (std, 1,2)
        return means, std


def main(args):
    if type(args) == dict:
        args = Namespace(**args)
    method = args.method
    seed_everything(args.seed, workers=True)

    log_dir = '{}/{}/{}/{}'.format(
        args.seed,
        args.save_dir,
        args.method,
        args.kernel_size,
    )

    logger = TensorBoardLogger(log_dir)

    if args.method == 'analytical':
        model = AnalyticalRegionAggregator(args)
    elif args.method == 'uniform':
        model = Uniform_model(args)
    
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
    parser.add_argument('--seed', type=int, default=80)
    parser.add_argument('--patience', type=int, default=100)

    parser.add_argument('--method', type=str, default='analytical')
    args = parser.parse_args()
    main(args)
