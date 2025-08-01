from numpy.core.numeric import indices
import torch
import pickle
import os
from tqdm import tqdm
import train 
import numpy as np
from dataset import Eurosat
from scipy import stats
import torch.distributions as dist
# This script is for getting some test metrics for MAE or MSE on the parcel value estimation

testset = Eurosat(mode='test')


def generate_pred_lists(model, dir_path, method):
    test_loader = torch.utils.data.DataLoader(testset,
                batch_size = 1,
                num_workers = 8,
                pin_memory=False
                ) 

    estimated_arr = []
    std_arr = []
    value_arr = []
    logs = []
    ros = []
    with torch.no_grad():
        for sample in tqdm(test_loader):

                images = sample['image']
                labels = sample['label']
                
                labels = model.avg_pool(labels)
                labels = torch.flatten(labels, 1, 2)

                estimated_values = model.pred_Out(images)[0]
                std_values = model.pred_Out(images)[1]
                
                estimated_values = model.avg_pool(estimated_values)
                std_values = model.avg_pool(std_values)

                estimated_values = torch.flatten(estimated_values, 1, 2)
                std_values = torch.flatten(std_values,1,2)

                
                estimated_arr.extend( estimated_values[0].cpu().numpy().tolist())
                std_arr.extend( std_values[0].cpu().numpy().tolist())
                value_arr.extend(labels[0].cpu().numpy().tolist())

                log = train.gaussLoss_train(estimated_values,torch.sqrt(std_values),labels).cpu().numpy().tolist()
                print(log)
                logs.append(log)

                gauss = dist.Normal (estimated_values, torch.sqrt(std_values))

                ro = ((gauss.cdf(labels + 0.8) - gauss.cdf(labels - 0.8))[0].cpu().numpy().tolist())
                ros.extend (ro)
                
                
                
    mse_errors = np.square( np.array(value_arr) - np.array(estimated_arr))
    log_error= (np.array(logs)).mean()
    ro_mean = (np.array(ros)).mean()
    
    #gaussian = ( np.array(value_arr) - np.array(model.gauss_fit()[0]))**2
    #print (gaussian.mean(), "gaussmodel mse") 
    print (np.array(std_arr).mean(), "std") 
    return mse_errors.mean(), log_error, ro_mean

def main(args):

    if type(args)==dict:
            args = Namespace(**args)
    dir_path = os.getcwd()
    test_file_path = os.path.join( dir_path, 'stats.txt')

    #use the desired check point path
    ckpt_path = os.path.join(dir_path, 'new_logs/rsample/8/70/default/version_0/checkpoints/epoch=48-step=8280.ckpt')
    torch.cuda.set_device(1)
    

    model = train.regionize_gauss(hparams = args)
    model = model.load_from_checkpoint(ckpt_path, 
        use_pretrained=False)
    mae_error,log_error, ro = generate_pred_lists(model, dir_path, args.method)
    
    test_file = open(test_file_path,"a")
    L = ["\nTest Stats: "+str(args.method), "\nMSE: "+ str(mae_error),
            "\nAverage Log Prob: "+ str(log_error),"\nro: "+ str(ro)   ]

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
