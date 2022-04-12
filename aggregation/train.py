# Requirements
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import util
from config import cfg
import modules
import os
from shutil import copyfile
import warnings

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

# Testing, small imports
from pytorch_lightning.callbacks import ModelCheckpoint
import time


def chooseModel(model_name = cfg.train.model):
    if model_name == "ral":
        model = modules.RALModule(use_pretrained=cfg.train.use_pretrained)
    if model_name == "uniform":
        model = modules.UniformModule(use_pretrained=cfg.train.use_pretrained)
        #cfg.data.sample_mode = 'uniform'
    if(model_name == 'rsample'):
        model = modules.RSampleModule(use_pretrained=cfg.train.use_pretrained)
    if(model_name == 'gauss'):
        model = modules.GaussModule(use_pretrained=cfg.train.use_pretrained)
    if(model_name == 'batch'):
        model = modules.BatchwiseMeanModule(use_pretrained=cfg.train.use_pretrained)
    return model

if __name__ == '__main__':
    model = chooseModel()
    train_loader, val_loader, test_loader = util.make_loaders()

    dir_path = os.path.join(os.getcwd(),'results', cfg.experiment_name)
    os.makedirs(dir_path, exist_ok=True)
    ckpt_path = os.path.join(dir_path,'best.ckpt')

    check_callback = ModelCheckpoint(monitor='val_loss', save_last=True, save_top_k=3, mode='min', filename='{epoch}-{val_loss:.2f}-{train_loss:.2f}')


    early_stopping = EarlyStopping('val_loss', patience=cfg.train.patience)

    
    trainer = pl.Trainer(gpus=cfg.train.device_ids, max_epochs = cfg.train.num_epochs, callbacks=[check_callback,early_stopping])
    t0 = time.time()
    trainer.fit(model, train_loader, val_loader)
    t1 = time.time()
    print('Training completed in', t1-t0, 'secs')
    print('Training completed in',(t1-t0)/60, 'mins')

    copyfile(check_callback.best_model_path, ckpt_path)

    train_file_path = os.path.join( dir_path, 'stats.txt')
    train_file = open(train_file_path,"a")
    L = ["Training Time: \n", str(t1-t0)+ " secs\n" , ""+ str((t1-t0)/60) + " mins\n" ]
    train_file.writelines(L)
    
