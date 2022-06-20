import torch
from dataset import Eurosat,Cifar
from tqdm import tqdm
import os
import train_new as train
import torch.distributions as dist
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

testset = Cifar(mode='test')

def gaussLoss_test(mean, std, target):
    gauss = dist.Normal(mean, std)
    loss = -gauss.log_prob(target)
    #loss = -(torch.mean(loss, 1))
    loss = torch.mean(loss)
    return loss

def generate_pred_lists(model, dir_path, method):
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=1,
                                              num_workers=8,
                                              pin_memory=True
                                              )

    estimated_arr = []
    std_arr = []
    value_arr = []
    logs = []
    ros15 = []
    ros25 = []
    ros35 = []
    ros50 = []
    ros1 = []
    i= 0
    with torch.no_grad():
        for sample in tqdm(test_loader):
            images = sample['image']

            labels = sample['label']
            labels = torch.flatten(labels, 1, 2)
    
            estimated_values, std_values = model.pred_Out(images)

            
            
           # estimated_values  = model.pred_Out(images)

            

            indices = labels.nonzero(as_tuple=True)
            labels = labels[indices]
            estimated_values = estimated_values[indices]
            std_values = std_values[indices]

            estimated_arr.extend(estimated_values.cpu().numpy().tolist())
            std_arr.extend(std_values.cpu().numpy().tolist())
            value_arr.extend(labels.cpu().numpy().tolist())

            log = gaussLoss_test(estimated_values, std_values, labels).cpu().numpy().tolist()
            logs.append(log)

            gauss = dist.Normal(estimated_values, std_values)

            ro15 = ((gauss.cdf(labels + 0.15) - gauss.cdf(labels - 0.15)).cpu().numpy().tolist())
            ro25 = ((gauss.cdf(labels + 0.25) - gauss.cdf(labels - 0.25)).cpu().numpy().tolist())
            ro35 = ((gauss.cdf(labels + 0.35) - gauss.cdf(labels - 0.35)).cpu().numpy().tolist())
            ro50 = ((gauss.cdf(labels + 0.5) - gauss.cdf(labels - 0.5)).cpu().numpy().tolist())
            ro1 = ((gauss.cdf(labels + 1.) - gauss.cdf(labels - 1.)).cpu().numpy().tolist())
            #print(ro15)
            ros15.extend(ro15)
            ros25.extend(ro25)
            ros35.extend(ro35)
            ros50.extend(ro50)
            ros1.extend(ro1)
            

           # print (labels.shape, labels.squeeze().shape)
           # print (estimated_values.shape, estimated_values.squeeze().shape)
           # print (images.shape, estimated_values.squeeze().shape)
            i += 1


    mse_errors = np.square(np.array(value_arr) - np.array(estimated_arr))
    log_error = (np.array(logs)).mean()
    ro15_mean = (np.array(ros15)).mean()
    ro25_mean = (np.array(ros25)).mean()
    ro35_mean = (np.array(ros35)).mean()
    ro50_mean = (np.array(ros50)).mean()
    ro1_mean = (np.array(ros1)).mean()
    print(np.array(std_arr).mean(), "std")
    
    value_arr = np.array(value_arr)
    #gaussian_MSE = ( value_arr - np.array(model.gauss_fit()[0]))**2
    #print (gaussian_MSE.mean(), "gauss_mse")
    #Gauss = model.gauss_fit()[2]
    #log_gaussian = (Gauss.log_prob (torch.tensor(value_arr))).mean()
    #print (log_gaussian,"gauss_log")
    #print (model.gauss_fit()[1], "gauss_std")
    #print (torch.mean(Gauss.cdf(torch.tensor(value_arr) + torch.tensor(0.5)) - Gauss.cdf(torch.tensor(value_arr) - torch.tensor(0.5))))
    #print (torch.mean(Gauss.cdf(torch.tensor(value_arr) + torch.tensor(0.15)) - Gauss.cdf(torch.tensor(value_arr) - torch.tensor(0.15))))
    return mse_errors.mean(), log_error, ro15_mean,ro25_mean,ro35_mean,ro50_mean,ro1_mean,np.array(std_arr).mean()


def main(args):
    if type(args) == dict:
        args = Namespace(**args)
    dir_path = os.getcwd()
    test_file_path = os.path.join(dir_path, 'logg.txt')

    # use the desired check point path
    ckpt_path = os.path.join(dir_path,
                             '/u/amo-d0/grad/cgar/Projects/disaggregation/synthetic_testing/80/logtest/analytical/16/10/lightning_logs/version_3/checkpoints/epoch=19-step=7820.ckpt')
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

    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=.01)
    parser.add_argument('--save_dir', default='logs')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=16)
    parser.add_argument('--patience', type=int, default=100)

    parser.add_argument('--method', type=str, default='analytical')

    args = parser.parse_args()
    main(args)

