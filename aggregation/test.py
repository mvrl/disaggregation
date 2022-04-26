from asyncio.unix_events import BaseChildWatcher
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
    pc_pth = os.path.join(dir_path, 'log.pkl')
    metric_pth = os.path.join(dir_path, 'metric.pkl')

    if (os.path.exists(est_pth)):
        with open(est_pth, "rb") as fp:
            estimated_arr = pickle.load(fp)
        with open(val_pth, "rb") as fp:
            value_arr = pickle.load(fp)
        with open(pc_pth, "rb") as fp:
            logs = pickle.load(fp)
        with open(metric_pth, "rb") as fp:
            metrics = pickle.load(fp)
    else:
        estimated_arr = []
        value_arr = []
        logs = []
        metrics = []

        print("Computing all predicted values...")
        with torch.no_grad():
            for batch in tqdm(test_loader):

                estimated_values, true_values = model.value_predictions(batch)
                #print(estimated_values.shape)
                print(estimated_values)
                print(true_values)

                if cfg.train.model == 'gauss' or cfg.train.model == 'rsample' or cfg.train.model == 'logsample':

                    interval_val = torch.tensor(10000)
                    log, metric = model.prob_eval(batch,interval_val)

                    logs.extend(log.cpu().numpy().tolist())
                    metrics.extend(metric.cpu().numpy().tolist())

                estimated_arr.extend( estimated_values.cpu().numpy().tolist())
                value_arr.extend(true_values.numpy().tolist())
                
    with open(est_pth, "wb") as fp:   #Pickling
        pickle.dump(estimated_arr, fp)
    with open(val_pth, "wb") as fp:   #Pickling
        pickle.dump(value_arr, fp)
    with open(pc_pth, "wb") as fp:   #Pickling
        pickle.dump(logs, fp)
    with open(metric_pth, "wb") as fp:   #Pickling
        pickle.dump(metrics, fp)

    mae_errors = np.abs( np.array(value_arr) - np.array(estimated_arr))
    relative_error = mae_errors / np.array(value_arr)
    mse_errors = np.array(value_arr) - np.array(estimated_arr)
    mse_errors = np.power(mse_errors,2)

    if cfg.train.model == 'gauss' or cfg.train.model == 'rsample' or cfg.train.model == 'logsample':
        logs= np.array(logs)
        metrics = np.array(metrics)
    else:
        logs= 0
        metrics = 0

    return mae_errors.mean(), mse_errors.mean(), np.mean(logs), relative_error.mean()*100, np.mean(metrics)

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

    mae_error, mse_error, log_error, percent_error, metric_mean = generate_pred_lists(model, dir_path)

    test_file = open(test_file_path,"a")
    L = ["\nTest Stats: ", "\nMAE: "+ str(mae_error) , "\nMSE: "+ str(mse_error), "\nAvg Percent Error: "+ str(percent_error), "\nAverage Log Prob: "+ str(log_error), "\nAverage Metric 10,0000 Probability: "+ str(metric_mean) ]
    test_file.writelines(L)
