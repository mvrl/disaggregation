import pytorch_lightning as pl
import torch 

from models import unet
import util

from collections import OrderedDict
import torch.nn as nn
from torchvision.utils import save_image


class aggregationModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.unet = unet.Unet(in_channels=3, out_channels=3)

        state_dict = torch.load('/u/pop-d1/grad/cgar222/Projects/disaggregation/building_segmentation/outputs/segpretrain/model_dict.pth')
        #Removing dictionary elements from nn.dataParrelel
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove module.
            new_state_dict[name] = v

        self.unet.load_state_dict(new_state_dict)

        for param in self.unet.parameters():
            param.requires_grad = False

        self.conv = nn.Conv2d(1,1,kernel_size = 1, padding = 0)
        self.softplus = nn.Softplus()
        self.agg = util.regionAgg_layer()


    def forward(self, x):
        x = self.unet(x)

        save_image(x[0][2], 'img1.png')
        x = self.conv(x[:][2].unsqueeze(1))
        x = self.softplus(x)

        save_image(x[0], 'img2.png')

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

        loss = util.regionAgg_loss(estimated_values, parcel_values)
        self.log('test_loss', loss)
        return loss

if __name__ == '__main__':

    train_loader, val_loader, test_loader = util.make_loaders()

    model = aggregationModule()
    trainer = pl.Trainer(gpus='0', max_epochs = 200)
    trainer.fit(model, train_loader, val_loader)
    
