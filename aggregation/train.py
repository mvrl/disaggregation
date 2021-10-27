# Requirements
import pytorch_lightning as pl
import util
from config import cfg
import modules

# Testing, small imports
from pytorch_lightning.callbacks import ModelCheckpoint
import time


def chooseModel(model_name = cfg.train.model):
    if model_name == "end2end":
        model = modules.RALModule(use_pretrained=False)
    if model_name == "pretrained":
        model = modules.RALModule(use_pretrained=True)
    if model_name == "uniform":
        model = modules.UniformModule(use_pretrained=True)
        cfg.train.uniform = True
    return model

if __name__ == '__main__':
    model = chooseModel()
    train_loader, val_loader, test_loader = util.make_loaders(uniform=cfg.uniform)

    #Init ModelCheckpoint callback, monitoring 'val_loss'

    check_callback = ModelCheckpoint(monitor='val_loss', save_last=True, save_top_k=5, mode='min', filename='{epoch}-{val_loss:.2f}-{train_loss:.2f}')
    ckpt_monitors = ()
    trainer = pl.Trainer(gpus=cfg.train.device_ids, max_epochs = cfg.train.num_epochs, callbacks=[check_callback])
    t0 = time.time()
    trainer.fit(model, train_loader, val_loader)
    t1 = time.time()
    print('Training completed in', t1-t0, 'secs')
    print('Training completed in',(t1-t0)/60, 'mins')
    
