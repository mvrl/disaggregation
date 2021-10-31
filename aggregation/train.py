# Requirements
import pytorch_lightning as pl
import util
from config import cfg
import modules
import os

# Testing, small imports
from pytorch_lightning.callbacks import ModelCheckpoint
import time


def chooseModel(model_name = cfg.train.model):
    if model_name == "end2end":
        model = modules.RALModule(use_pretrained=False)
    if model_name == "pretrained":
        model = modules.RALModule(use_pretrained=True) # this can be a config setting
    if model_name == "uniform":
        model = modules.UniformModule(use_pretrained=True)
        cfg.data.sample_mode = 'uniform'
    if(model_name == 'agg'):
        model = modules.AggregatedModule(use_pretrained=False)
        cfg.data.sample_mode ='agg'
    return model

if __name__ == '__main__':
    model = chooseModel()
    train_loader, val_loader, test_loader = util.make_loaders()

    dir_path = os.path.join(os.getcwd(),'results', cfg.experiment_name)
    os.makedirs(dir_path, exist_ok=True)

    check_callback = ModelCheckpoint(monitor='val_loss', save_last=True, save_top_k=3, mode='min', filename='{epoch}-{val_loss:.2f}-{train_loss:.2f}')

    trainer = pl.Trainer(default_root_dir = dir_path,gpus=cfg.train.device_ids, max_epochs = cfg.train.num_epochs, callbacks=[check_callback])
    t0 = time.time()
    trainer.fit(model, train_loader, val_loader)
    t1 = time.time()
    print('Training completed in', t1-t0, 'secs')
    print('Training completed in',(t1-t0)/60, 'mins')

    train_file_path = os.path.join( dir_path, 'stats.txt')
    train_file = open(train_file_path,"a")
    L = ["Training Time: \n", str(t1-t0)+ " secs\n" , ""+ str((t1-t0)/60) + " mins\n" ]
    train_file.writelines(L)
    
