from numpy.core.numeric import indices
import util
import torch
import pickle
import os
from tqdm import tqdm
from config import cfg
import modules
import numpy as np
import train
# This script is for getting some test metrics for MAE or MSE on the parcel value estimation

# Move this to a bonafide testing script
def generate_pred_lists(model, dir_path):
    train_loader, val_loader, test_loader = util.make_loaders(batch_size = 1, mode = 'test', sample_mode='')

    est_pth = os.path.join(dir_path, 'estimated.pkl')
    val_pth = os.path.join(dir_path, 'value.pkl')
    pc_pth = os.path.join(dir_path, 'perchip.pkl')

    if (os.path.exists(est_pth)):
        with open(est_pth, "rb") as fp:\
            estimated_arr = pickle.load(fp)
        with open(val_pth, "rb") as fp:
            value_arr = pickle.load(fp)
        with open(pc_pth, "rb") as fp:
            per_chip_errors = pickle.load(fp)
    else:
        estimated_arr = []
        value_arr = []
        per_chip_errors = []

        print("Computing all predicted values...")
        with torch.no_grad():
            for sample in tqdm(test_loader):
                image, mask, value = sample

                estimated_values = model.pred_Out(image, mask)

                estimated_arr.extend( estimated_values[0].cpu().numpy().tolist())
                value_arr.extend(value[0].numpy().tolist())

                per_chip_error = estimated_values[0].cpu().numpy().sum() - value[0].numpy().sum()
                per_chip_errors.append(per_chip_error)

    with open(est_pth, "wb") as fp:   #Pickling
        pickle.dump(estimated_arr, fp)
    with open(val_pth, "wb") as fp:   #Pickling
        pickle.dump(value_arr, fp)
    with open(pc_pth, "wb") as fp:   #Pickling
        pickle.dump(per_chip_errors, fp)

    mae_errors = np.abs( np.array(value_arr) - np.array(estimated_arr))
    mse_errors = np.array(value_arr) - np.array(estimated_arr)
    mse_errors = np.power(mse_errors,2)
    per_chip_error = np.abs(np.array(per_chip_errors)).mean()

    return mae_errors.mean(), mse_errors.mean(), per_chip_error

def loadModel(ckpt_path, model_name = cfg.train.model):
    model =train.chooseModel(model_name)

    model = model.load_from_checkpoint(ckpt_path, 
        use_pretrained=False)
    return model

if __name__ == '__main__':
    dir_path = os.path.join(os.getcwd(), 'results', cfg.experiment_name)
    ckpt_path = os.path.join(dir_path,'best.ckpt')
    test_file_path = os.path.join( dir_path, 'stats.txt')

    torch.cuda.set_device(1)
    
    model = loadModel(ckpt_path , cfg.train.model)

    mae_error, mse_error, per_chip_error = generate_pred_lists(model, dir_path)

    test_file = open(test_file_path,"a")
    L = ["\nTest Stats: ", "\nMAE: "+ str(mae_error) , "\nMSE: "+ str(mse_error), "\nPer Chip MAE: "+ str(per_chip_error) ]
    test_file.writelines(L)
