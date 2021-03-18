# training script

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from config import cfg

from net_factory import get_network
from data_factory import get_data

def save_batch_images(source, seg_predictions, seg_labels,reg_predictions,reg_labels,out_dir,ctr):
    # normalize the prediction
    seg_predictions = F.softmax(seg_predictions, dim=1)
    reg_predictions = F.softmax(reg_predictions, dim=1)       # for training, PyTorch expects the prediction to be unnormalized
    
    n_classes = seg_predictions.shape[1]

    seg_predictions = torch.argmax(seg_predictions, dim=1)

    reg_predictions = torch.argmax(reg_predictions, dim=1)

    batch_n = seg_predictions.shape[0]

    plt.ioff()
    for k in range(batch_n):
        plt.figure()
        img_now = source[k, :, :, :].permute(1,2,0)
        plt.imshow(img_now.detach().cpu().numpy())
        plt.axis('off')
        fname1 = str(str(ctr) +'_' +str(k) + '_input' + '.png')
        plt.savefig(os.path.join(out_dir, fname1), bbox_inches='tight')
        plt.close()

        plt.figure()

        plt.imshow(seg_predictions[k, :, :].detach().cpu().numpy())
        plt.axis('off')
        fname1 = str(str(ctr) +'_' +str(k) + '_SEGpred' + '.png')
        plt.savefig(os.path.join(out_dir, fname1), bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.imshow(seg_labels[k, :, :].detach().cpu().numpy())
        plt.axis('off')
        fname1 = str(str(ctr) +'_' +str(k) + '_SEGtarget' + '.png')
        plt.savefig(os.path.join(out_dir, fname1), bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.imshow(reg_predictions[k, :, :].detach().cpu().numpy())
        plt.axis('off')
        fname1 = str(str(ctr) +'_' +str(k) + '_REGpred' + '.png')
        plt.savefig(os.path.join(out_dir, fname1), bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.imshow(reg_labels[k, :, :].detach().cpu().numpy())
        plt.axis('off')
        fname1 = str(str(ctr) +'_' +str(k) + '_REGtarget' + '.png')
        plt.savefig(os.path.join(out_dir, fname1), bbox_inches='tight')
        plt.close()
        
        
        plt.figure(dpi=300)
        plt.subplot(3,2,1)
        plt.imshow(img_now.detach().cpu().numpy())
        plt.axis('off')

        plt.subplot(3,2,2)
        plt.imshow(seg_predictions[k, :, :].detach().cpu().numpy())
        #plt.colorbar()
        plt.title('seg_prediction')
        plt.axis('off')

        plt.subplot(3,2,4)
        plt.imshow(reg_predictions[k, :, :].detach().cpu().numpy())
        #plt.colorbar()
        plt.title('reg_prediction')
        plt.axis('off')

        plt.subplot(3,2,5)
        plt.imshow(reg_labels[k, :, :].detach().cpu().numpy())
        #plt.colorbar()
        plt.title('reg_GT')
        plt.axis('off')

        plt.subplot(3,2,3)
        plt.imshow(seg_labels[k, :, :].detach().cpu().numpy())
        plt.axis('off')
        plt.title('seg_GT')
        #plt.colorbar()
        fname1 = str(str(ctr) +'_' +str(k) + '_combined' + '.png')
        plt.savefig(os.path.join(out_dir, fname1), bbox_inches='tight')
        plt.close()


##=========================================
## Setup

    
def main():
    
    which_model = 'best'            # which model checkpoint to use. options: 'best' or 'end'

    folder_name = 'seg_results'

    out_dir = cfg.train.out_dir
    
    # check if the trained model directory exists
    if not os.path.exists(out_dir):
        raise ValueError('The directory with trained model does not exist... make sure cfg.train.out_dir in config.py has the correct directory name')
    
    full_dir_name = os.path.join(out_dir, folder_name)
    if os.path.exists(full_dir_name):
        raise ValueError(
            'The validation folder image_results already exists. Delete the folder if those results are not needed')
    else:
        os.makedirs(full_dir_name)

    ## Get the model
    model = get_network(cfg.model.name)
    fname = os.path.join(out_dir, 'model_dict.pth')

    model.load_state_dict(torch.load(fname))
    model.eval()
    print('Network loaded....')

    # Datasets
    eval_mode = 'test'
    data_loader = get_data(cfg, mode=eval_mode)

   
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            if i > 2:       # save this many batches
                break
                
            # reading data
            image = data[0].cuda()
            seg_labels = data[1].long().cuda()
            reg_labels = data[2].float().cuda()

            seg_predictions, reg_predictions = model(image)
            if i==0:
                print('predictions size: ', seg_predictions.shape)

            save_batch_images(image, seg_predictions, seg_labels,reg_predictions,reg_labels, full_dir_name, i)

    print('All done!')

if __name__ == '__main__':
    main()    
