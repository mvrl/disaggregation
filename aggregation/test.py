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

    path = os.path.join(dir_path, 'test_results')
    if not os.path.exists(path):
        folder = os.makedirs(path)

    est_pth = os.path.join(path, 'estimated.pkl')
    val_pth = os.path.join(path, 'value.pkl')
    pc_pth = os.path.join(path, 'log.pkl')
    metric10k_pth = os.path.join(path, 'metric10k.pkl')
    metric100k_pth = os.path.join(path, 'metric100k.pkl')
    stds_pth = os.path.join(path, 'stds.pkl')

    if (os.path.exists(est_pth)):
        with open(est_pth, "rb") as fp:
            estimated_arr = pickle.load(fp)
        with open(val_pth, "rb") as fp:
            value_arr = pickle.load(fp)
        with open(pc_pth, "rb") as fp:
            logs = pickle.load(fp)
        with open(metric10k_pth, "rb") as fp:
            metrics10k = pickle.load(fp)
        with open(metric100k_pth, "rb") as fp:
            metrics100k = pickle.load(fp)
        with open(stds_pth, "rb") as fp:
            stds = pickle.load(fp)
    else:
        estimated_arr = []
        value_arr = []
        logs = []
        metrics10k = []
        metrics100k = []
        stds = []

        model.eval()

        print("Computing all predicted values...")
        with torch.no_grad():
            for batch in tqdm(test_loader):

                estimated_values, true_values = model.value_predictions(batch)
                #print(estimated_values.shape)
                print(estimated_values)
                print(true_values)

                if cfg.train.model == 'gauss' or cfg.train.model == 'rsample' or cfg.train.model == 'logsample':

                    interval_val1 = torch.tensor(10000)
                    interval_val2 = torch.tensor(100000)
                    log, metric10k, metric100k, std = model.prob_eval(batch, interval_val1, interval_val2)

                    logs.extend(log.cpu().numpy().tolist())
                    metrics10k.extend(metric10k.cpu().numpy().tolist())
                    metrics100k.extend(metric100k.cpu().numpy().tolist())
                    stds.extend(std.cpu().numpy().tolist())

                estimated_arr.extend( estimated_values.cpu().numpy().tolist())
                value_arr.extend(true_values.numpy().tolist())
                
    with open(est_pth, "wb") as fp:   #Pickling
        pickle.dump(estimated_arr, fp)
    with open(val_pth, "wb") as fp:   #Pickling
        pickle.dump(value_arr, fp)
    with open(pc_pth, "wb") as fp:   #Pickling
        pickle.dump(logs, fp)
    with open(metric10k_pth, "wb") as fp:   #Pickling
        pickle.dump(metrics10k, fp)
    with open(metric100k_pth, "wb") as fp:   #Pickling
        pickle.dump(metrics100k, fp)
    with open(stds_pth, "wb") as fp:   #Pickling
        pickle.dump(stds, fp)

    mae_errors = np.abs( np.array(value_arr) - np.array(estimated_arr))
    relative_error = mae_errors / np.array(value_arr)
    mse_errors = np.array(value_arr) - np.array(estimated_arr)
    mse_errors = np.power(mse_errors,2)
    

    if cfg.train.model == 'gauss' or cfg.train.model == 'rsample' or cfg.train.model == 'logsample':
        logs= np.array(logs)
        metrics10k = np.array(metrics10k)
        metrics100k = np.array(metrics100k)
        stds = np.array(stds)
    else:
        logs= 0
        metrics10k = 0
        metrics100k = 0

    return mae_errors.mean(), mse_errors.mean(), np.mean(logs), np.median(logs), relative_error.mean()*100, np.mean(metrics10k), np.mean(metrics100k), np.mean(stds), np.mean(np.exp(logs))

def loadModel(ckpt_path, model_name = cfg.train.model):
    model =train.chooseModel(model_name)
    if(model_name == 'logsample'):
         model = model.load_from_checkpoint(ckpt_path, 
        use_pretrained=False, num_samples= cfg.train.num_samples)
    else:
        model = model.load_from_checkpoint(ckpt_path, 
        use_pretrained=False)
    return model

if __name__ == '__main__':
    dir_path = os.path.join(os.getcwd(), 'results', cfg.experiment_name)
    ckpt_path = os.path.join(dir_path,'best.ckpt')
    test_file_path = os.path.join( dir_path, 'stats.txt')

    torch.cuda.set_device(1)

    model = loadModel(ckpt_path , cfg.train.model)

    mae_error, mse_error, log_error, med_log_error, percent_error, metric10k_mean, metric100k_mean, variances_mean, pdf_mean = generate_pred_lists(model, dir_path)

    test_file = open(test_file_path,"a")
    L = ["\nTest Stats: ", "\nMAE: "+ str(mae_error) , "\nMSE: "+ str(mse_error), "\nAvg Percent Error: "+ str(percent_error),"\nMean Probability: "+ str(pdf_mean), "\nAverage Log Prob: "+ str(log_error),"\nMedian Log Prob: "+ str(med_log_error),
     "\nAverage Metric 10,0000 Probability: "+ str(metric10k_mean), "\nAverage Metric 100,0000 Probability: "+ str(metric100k_mean), "\nAverage Predicted Standard Deviation: "+ str(variances_mean) ]
    test_file.writelines(L)
