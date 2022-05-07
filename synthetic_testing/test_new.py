import torch
from dataset import Eurosat
from tqdm import tqdm
import os
import train_new as train
import torch.distributions as dist
import matplotlib.pyplot as plt
import numpy as np

testset = Eurosat(mode='test')


def generate_pred_lists(model, dir_path, method):
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=1,
                                              num_workers=8,
                                              pin_memory=False
                                              )

    estimated_arr = []
    std_arr = []
    value_arr = []
    logs = []
    ros = []
    # i= 0
    with torch.no_grad():
        for sample in tqdm(test_loader):
            images = sample['image']

            labels = sample['label']
            labels = torch.flatten(labels, 1, 2)

            estimated_values = model.pred_Out(images)[0]
            std_values = model.pred_Out(images)[1]

            estimated_arr.extend(estimated_values[0].cpu().numpy().tolist())
            std_arr.extend(std_values[0].cpu().numpy().tolist())
            value_arr.extend(labels[0].cpu().numpy().tolist())

            log = train.gaussLoss_test(estimated_values, std_values, labels).cpu().numpy().tolist()
            print(log)
            logs.append(log)

            gauss = dist.Normal(estimated_values, model.pred_Out(images)[1])

            ro = ((gauss.cdf(labels + 0.8) - gauss.cdf(labels - 0.8))[0].cpu().numpy().tolist())
            ros.extend(ro)

    # plt.imsave( "label"+str(i)+".png", (torch.tensor(labels.squeeze())).cpu())
    # plt.imsave("mean"+str(i)+".png",(torch.tensor((estimated_values.squeeze()))).cpu())
    # i += 1


    mse_errors = np.square(np.array(value_arr) - np.array(estimated_arr))
    log_error = (np.array(logs)).mean()
    ro_mean = (np.array(ros)).mean()

    print(np.array(std_arr).mean(), "std")
    return mse_errors.mean(), log_error, ro_mean


def main(args):
    if type(args) == dict:
        args = Namespace(**args)
    dir_path = os.getcwd()
    test_file_path = os.path.join(dir_path, 'stats.txt')

    # use the desired check point path
    ckpt_path = os.path.join(dir_path,
                             'new_logs/analytical/16/10/default/version_22/checkpoints/epoch=14-step=2849.ckpt')
    torch.cuda.set_device(1)
    if args.method == 'analytical':
        model = train.AnalyticalRegionAggregator(args)
        model = model.load_from_checkpoint(ckpt_path,
                                           use_pretrained=False)
    elif args.method == 'rsample':
        model = train.SamplingRegionAggregator(10, args)
        model = model.load_from_checkpoint(ckpt_path,
                                           use_pretrained=False, k=10)

    mae_error, log_error, ro = generate_pred_lists(model, dir_path, args.method)

    test_file = open(test_file_path, "a")
    L = ["\nTest Stats 100: " + str(args.method), "\nMSE: " + str(mae_error),
         "\nAverage Log Prob: " + str(log_error), "\nro: " + str(ro)]

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

