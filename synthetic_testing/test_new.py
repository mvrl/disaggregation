import torch
from dataset import Eurosat
from tqdm import tqdm
import os
import train_new as train
import torch.distributions as dist
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd

testset = Eurosat(mode='test')

def generate_pred_lists(model, dir_path, method):
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=1,
                                              num_workers=8,
                                              pin_memory=False
                                              )
    
    means, y_true, mse_errors, stds, logs, ros15, ros25, ros35, ros50, ros1 = ([] for i in range(10))
    i= 0
    with torch.no_grad():
        for sample in tqdm(test_loader):
            images = sample['image']

            labels = sample['label']

            mu_values, std_values = model.pred_Out(images)

            gauss = dist.Normal(mu_values, std_values)
            log = gauss.log_prob(labels)
            logs.extend(log[0].cpu().numpy().tolist())

            ros15.extend((gauss.cdf(labels + 0.15) - gauss.cdf(labels - 0.15))[0].cpu().numpy().tolist())
            ros25.extend((gauss.cdf(labels + 0.25) - gauss.cdf(labels - 0.25))[0].cpu().numpy().tolist())
            ros35.extend((gauss.cdf(labels + 0.35) - gauss.cdf(labels - 0.35))[0].cpu().numpy().tolist())
            ros50.extend((gauss.cdf(labels + 0.5) - gauss.cdf(labels - 0.5))[0].cpu().numpy().tolist())
            ros1.extend((gauss.cdf(labels + 1.) - gauss.cdf(labels - 1.))[0].cpu().numpy().tolist())

            mse_errors.extend(((labels-mu_values)**2)[0].cpu().numpy().tolist())
            means.extend(mu_values[0].cpu().numpy().tolist())
            stds.extend(std_values[0].cpu().numpy().tolist())
            y_true.extend(labels[0].cpu().numpy().tolist())
    
            i += 1

    #cols=['mean','y_true','mse','std','log_prob','ros15','ros25','ros35','ros50','ros1']
    #df = pd.DataFrame({'mean':means,'y_true': y_true, 'mse':mse_errors, 'std':stds, 'log_prob':logs, 'ros15':ros15,'ros25': ros25,'ros35': ros35, 'ros50':ros50,'ros1': ros1},columns=cols)
    #print(df)
    #print(df['std'].mean())

    plt.hist( np.array(ros50), bins=100)
    plt.savefig('plot.jpg')
    plt.show()

    plt.hist( np.array(logs), bins=100)
    plt.xlim([-4,2])
    plt.savefig('plot2.jpg')
    plt.show()

    mse_error = (np.array(mse_errors)).mean()
    log_error = (np.array(logs)).mean()
    ro15_mean = (np.array(ros15)).mean()
    ro25_mean = (np.array(ros25)).mean()
    ro35_mean = (np.array(ros35)).mean()
    ro50_mean = (np.array(ros50)).mean()
    ro1_mean = (np.array(ros1)).mean()
    std_mean = (np.array(stds)).mean()
    
    return mse_error,log_error, ro15_mean,ro25_mean,ro35_mean,ro50_mean,ro1_mean,std_mean


def main(args):
    if type(args) == dict:
        args = Namespace(**args)
    dir_path = os.getcwd()
    test_file_path = os.path.join(dir_path, 'logg.txt')

    # use the desired check point path
    ckpt_path = os.path.join(dir_path,
                             '/u/amo-d0/grad/cgar/Projects/disaggregation/synthetic_testing/80/logtest/analytical/8/10/lightning_logs/version_28/checkpoints/epoch=147-step=28120.ckpt')
    torch.cuda.set_device(1)
    if args.method == 'analytical':
        model = train.AnalyticalRegionAggregator(args)
        model = model.load_from_checkpoint(ckpt_path,
                                           use_pretrained=False).eval()
    elif args.method == 'rsample':
        model = train.SamplingRegionAggregator(10, args)
        model = model.load_from_checkpoint(ckpt_path,
                                           use_pretrained=False, k=10).eval()
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
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=.01)
    parser.add_argument('--save_dir', default='logs')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=16)
    parser.add_argument('--patience', type=int, default=100)

    parser.add_argument('--method', type=str, default='analytical')

    args = parser.parse_args()
    main(args)

