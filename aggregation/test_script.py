import trainAgg
import util
from PIL import Image
import torch
from torchvision import transforms
import pytorch_lightning as pl
import matplotlib as pyplot

def test(train_loader):
    # initialize 4 of the region agg networks/modules

    # they should have random initialization on the 4 random weights
    model1 = trainAgg.aggregationModule()
    model2 = trainAgg.aggregationModule()
    model3 = trainAgg.aggregationModule()

    trainer = pl.Trainer(gpus='0', max_epochs = 50)

    #Train them 
    trainer.fit(model1,train_loader)
    trainer2 = pl.Trainer(gpus='0', max_epochs = 50)
    trainer2.fit(model2,train_loader)
    trainer3 = pl.Trainer(gpus='0', max_epochs = 50)
    trainer3.fit(model3,train_loader)

if __name__ == '__main__':

    # we can also just take an index from the loaders but that takes forever
    train_loader, val_loader, test_loader = util.make_loaders()

    test(train_loader)