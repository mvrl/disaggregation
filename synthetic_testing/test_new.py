import torch
from dataset import Eurosat
from tqdm import tqdm
import os
import train_small as train
import torch.distributions as dist
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import norm
import statistics

testset = Eurosat(mode='test')


def generate_pred_lists(model, dir_path, method):
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=1,
                                              num_workers=8,
                                              pin_memory=False
                                              )

    means,stds,targets,logs, ros15,ros25,ros35,ros50,ros1 = ([] for i in range(9))
    
    with torch.no_grad():
        for sample in tqdm(test_loader):

            images = sample['image']
            labels = sample['label']
            labels = torch.flatten(labels, 1, 2)

            means_values, stds_values = model.pred_Out(images)

            means.extend(means_values[0].cpu().numpy().tolist())
            stds.extend(stds_values[0].cpu().numpy().tolist())
            targets.extend(labels[0].cpu().numpy().tolist())

            log = train.gaussLoss(means_values, stds_values, labels).cpu().numpy().tolist()
           # print(log)
            logs.append(log)

            gauss = dist.Normal(means_values, stds_values)

            ros15.extend ((gauss.cdf(labels + 0.15) - gauss.cdf(labels - 0.15))[0].cpu().numpy().tolist())
            ros25.extend ((gauss.cdf(labels + 0.25) - gauss.cdf(labels - 0.25))[0].cpu().numpy().tolist())
            ros35.extend ((gauss.cdf(labels + 0.35) - gauss.cdf(labels - 0.35))[0].cpu().numpy().tolist())
            ros50.extend ((gauss.cdf(labels + 0.5) - gauss.cdf(labels - .5))[0].cpu().numpy().tolist())
            ros1.extend ((gauss.cdf(labels + 1.) - gauss.cdf(labels - 1.))[0].cpu().numpy().tolist())
            
    mse_errors = np.square(np.array(targets) - np.array(means))
    log_error = (np.array(logs)).mean()
    ro15_mean = (np.array(ros15)).mean()
    ro25_mean = (np.array(ros25)).mean()
    ro35_mean = (np.array(ros35)).mean()
    ro50_mean = (np.array(ros50)).mean()
    ro1_mean = (np.array(ros1)).mean()
    
    value_arr = np.array(targets)
    return mse_errors.mean(), log_error, ro15_mean,ro25_mean,ro35_mean,ro50_mean,ro1_mean,np.array(stds).mean()


def main(args):
    if type(args) == dict:
        args = Namespace(**args)
    dir_path = os.getcwd()
    test_file_path = os.path.join(dir_path, 'logg.txt')

    # use the desired check point path
    ckpt_path = os.path.join(dir_path,
                             '80/logtest/uniform/16/default/version_2/checkpoints/epoch=66-step=12729.ckpt')
    torch.cuda.set_device(1)
    if args.method == 'analytical':
        model = train.AnalyticalRegionAggregator(args)
        model = model.load_from_checkpoint(ckpt_path,
                                           use_pretrained=False).eval()
    elif args.method == 'uniform':
        model = train.Uniform_model( args)
        model = model.load_from_checkpoint(ckpt_path,
                                           use_pretrained=False).eval()

    mse_error, log_error, ro15,ro25,ro35,ro50,ro1,std = generate_pred_lists(model, dir_path, args.method)

    test_file = open(test_file_path, "a")
    L = ["\nStats 10: " + str(args.method), "\nMSE: " + str(mse_error),
         "\nAverage Log Prob: " + str(log_error), "\nro15: " + str(ro15),"\nro25: " + str(ro25), 
         "\nro35: " + str(ro35),"\nro50: " + str(ro50),"\nro1: " + str(ro1),"\nstd: " + str(std)]

    test_file.writelines(L)


if __name__ == '__main__':
    from argparse import ArgumentParser, Namespace

    parser = ArgumentParser()

    parser.add_argument('--max_epochs', type=int, default=150)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=.0001)
    parser.add_argument('--save_dir', default='logs')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=16)
    parser.add_argument('--patience', type=int, default=100)

    parser.add_argument('--method', type=str, default='analytical')

    args = parser.parse_args()
    main(args)

