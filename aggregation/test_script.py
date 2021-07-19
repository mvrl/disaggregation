from numpy.core.numeric import indices
import trainAgg
import util
from PIL import Image
import torch
from torchvision import transforms
import pytorch_lightning as pl
import matplotlib as pyplot
import numpy as np

def test(self, test_loader):

    #trainer = pl.Trainer(gpus='0', max_epochs = 50)

    individual_errors = []

    with torch.no_grad():
        for batch in test_loader:
            mse = self.test_step(batch)
            individual_errors.append(mse)

    individual_errors = np.array(individual_errors)

    return individual_errors

if __name__ == '__main__':

    trainer = pl.Trainer()
    trainer = pl.Trainer(gpus='0', max_epochs = 200)

    # we can also just take an index from the loaders but that takes forever
    train_loader, val_loader, test_loader = util.make_loaders(batch_size = 1)

    model = trainAgg.aggregationModule(use_pretrained=False)
    model = model.load_from_checkpoint('/u/pop-d1/grad/cgar222/Projects/disaggregation/aggregation/lightning_logs/version_135/checkpoints/epoch=104-step=43889.ckpt', use_pretrained=False)

    trainer.test(model, test_loader)

    #print(np.mean())

    #np.save('new_test', errors )
    # for making graphs tomorrow
