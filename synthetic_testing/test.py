from numpy.core.numeric import indices
import torch
import pickle
import os
from tqdm import tqdm
import train 
import numpy as np
from dataset import Eurosat

# This script is for getting some test metrics for MAE or MSE on the parcel value estimation

testset = Eurosat(mode='test')


def generate_pred_lists(model, dir_path, method):
    test_loader = torch.utils.data.DataLoader(testset,
                batch_size = 1,
                num_workers = 8,
                pin_memory=False
                ) 

    estimated_arr = []
    value_arr = []
    logs = []

    with torch.no_grad():
        for sample in tqdm(test_loader):
                
                images = sample['image']
                labels = sample['label']

                
                estimated_values = model.pred_Out(images)
                
                print ("size of est", estimated_values.size())

                estimated_arr.extend( estimated_values[0].cpu().numpy().tolist())
                value_arr.extend(labels[0].cpu().numpy().tolist())

                
                log = model.log_out(images,labels)[0].cpu().numpy().tolist()
                logs.extend(log)

    mae_errors = np.abs( np.array(value_arr) - np.array(estimated_arr))
    log_error= np.abs(np.array(logs)).mean()

    return mae_errors.mean(), log_error

def main(args):

    if type(args)==dict:
            args = Namespace(**args)
    dir_path = os.getcwd()
    test_file_path = os.path.join( dir_path, 'stats.txt')

    #use the desired check point path
    ckpt_path = os.path.join(dir_path, 'logs/rsample/16/default/version_0/checkpoints/last.ckpt')
    torch.cuda.set_device(1)
    

    model = train.regionize_gauss(hparams = args)
    model = model.load_from_checkpoint(ckpt_path, 
        use_pretrained=False)
    mae_error,log_error = generate_pred_lists(model, dir_path, args.method)
    
    test_file = open(test_file_path,"a")
    L = ["\nTest Stats for: "+str(args.method), "\nMAE: "+ str(mae_error),
            "\nAverage Log Prob: "+ str(log_error)   ]

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
