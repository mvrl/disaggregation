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

def get_confusion_matrix(label, pred, size, num_class, ignore=-1, use_softmax = False):
    """
    Calcute the confusion matrix by given label and pred
    source: https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/pytorch-v1.1/lib/utils/utils.py
    """
    if use_softmax:
        pred = F.softmax(pred, dim=1)
    
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,i_pred] = label_count[cur_index]
    return confusion_matrix

##=========================================
## Setup

    
def main():
    
    which_model = 'best'            # which model checkpoint to use. options: 'best' or 'end'

    folder_name = 'eval_results'

    out_dir = cfg.train.out_dir
    
    # check if the trained model directory exists
    if not os.path.exists(out_dir):
        raise ValueError('The directory with trained model does not exist... make sure cfg.train.out_dir in config.py has the correct directory name')
    
    full_dir_name = os.path.join(out_dir, folder_name)
    if os.path.exists(full_dir_name):
        print('The eval folder image_results already exists. Overwriting results')
    else:
        os.makedirs(full_dir_name)

    # get the model
    model = get_network(cfg.model.name)
    fname = os.path.join(out_dir, 'model_dict.pth')

    model.load_state_dict(torch.load(fname))
    model.eval()
    print('Network loaded....')

    # get data
    eval_mode = cfg.data.eval_mode
    data_loader = get_data(cfg, mode=eval_mode)
    
    num_classes = cfg.model.out_channels
    confusion_matrix = np.zeros((num_classes, num_classes))
   
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            # reading data
            image = data[0].cuda()
            labels = data[1].cuda().long()

            predictions = model(image)

            confusion_matrix += get_confusion_matrix(label=labels, pred=predictions,size=cfg.data.cutout_size,num_class=num_classes, use_softmax=True)

    
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    print('results on split:', eval_mode)
    print('pixel acc: ', pixel_acc)
    print('mean acc: ', mean_acc)
    print('IoU full: ', IoU_array)
    print('m IoU: ', mean_IoU)

    # Saving logs
    name_string = 'eval_result_' + eval_mode +'.txt'
    fname = os.path.join(out_dir, name_string)
    with open(os.path.join(fname), 'w') as result_file:
        result_file.write('Pixel acc ')
        result_file.write(str(pixel_acc))
        result_file.write('\nMean Acc  ')
        result_file.write(str(mean_acc))
        result_file.write('\nIoU full ')
        result_file.write(str(IoU_array))
        result_file.write('\nmean Iou ')
        result_file.write(str(mean_IoU))
    
    print('finished evaluation!')

if __name__ == '__main__':
    main()    
