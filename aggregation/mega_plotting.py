from cmath import nan
import matplotlib.pyplot as plt
import util
import torch
import os
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import pickle
import seaborn as sns
from config import cfg
import test

def generate_plot(image,vals, pred_map, true_map, region_map, path):
    fig, axs = plt.subplots(4,1,figsize=(3,12))
    axs[0].imshow(image.permute(1,2,0) )
    axs[0].axis('off')

    axs[1].imshow(region_map , cmap = 'tab20')
    axs[1].axis('off')
    #axs[0].set_title("Image")
    print(vals.shape)
    axs[2].imshow(vals , cmap = 'Greens')
    axs[2].axis('off')
    #axs[0].set_title("Value Prediction")
    # CRATING AVERAGE VALUE MASK COMBINATION
    axs[3].imshow(pred_map, cmap = 'Greens')
    axs[3].axis('off')
    #axs[1][1].set_title("Aggregated Value Prediction")

    # This needs proper color scalings... unsure how to do this
    #axs[1][0].imshow(true_map, cmap = 'Greens')
    #axs[1][0].set_title("True Value Map")
    #axs[1][0].axis('off')
    fig.tight_layout(pad=0)
    plt.savefig(path)
    plt.close(fig)

def generate_images(model, num_images, dir_path):
    train_loader, val_loader, test_loader = util.make_loaders(batch_size = 1, mode = 'vis', sample_mode='')

    c = 0
    with torch.no_grad():
        for sample in test_loader:
            image, masks, value = sample

            value = value[0]

            vals = model.get_valOut(image)

            estimated_values = model.pred_Out(image, masks)
            #estimated_values = estimated_values.cpu().detach().numpy()
            # CRATING AVERAGE VALUE MASK COMBINATION
            masks = masks[0]
            estimated_values = estimated_values[0]
            pred_map = np.zeros_like(masks[0])
            region_map = np.zeros_like(masks[0])
            mask_sum= np.zeros_like(masks[0])

            for i,mask in enumerate(masks):
                mask = np.array(mask)
                region_map = np.add(mask*np.random.randint(0,100), region_map)

            region_map = region_map.reshape(cfg.data.cutout_size).astype('float64')
            region_map[region_map==0] = nan

            for i,mask in enumerate(masks):
                mask = np.array(mask)
                mask_sum += mask
                pixel_count = (mask == 1).sum()
                uniform_value = estimated_values[i]/pixel_count
                uniform_value = uniform_value.cpu().detach().numpy()
                pred_map = np.add(mask*uniform_value, pred_map)
            
            pred_map = pred_map.reshape(cfg.data.cutout_size)

            true_map = np.zeros_like(masks[0])
            for i,mask in enumerate(masks):
                mask = np.array(mask)
                pixel_count = (mask == 1).sum()
                uniform_value = value[i]/pixel_count
                uniform_value = uniform_value.cpu().detach().numpy()
                true_map = np.add(mask*uniform_value, true_map)
            
            true_map = true_map.reshape(cfg.data.cutout_size)

            vals = vals.squeeze(0).squeeze(0).cpu().detach().numpy() #* mask_sum.reshape(512,512)

            path =  os.path.join(dir_path, str(c) )

            generate_plot(image.squeeze(0),vals,pred_map,true_map, region_map, path)
            c+=1
            if c >= num_images:
                return


if __name__ == '__main__':
    
    dir_path = os.path.join(os.getcwd(), 'results', cfg.experiment_name)
    vis_path= os.path.join(dir_path ,'visualizations_diff/')
    ckpt_path = os.path.join(dir_path,'best.ckpt')

    if not(os.path.exists(vis_path)):
        os.mkdir(vis_path)

    model = test.loadModel(ckpt_path , cfg.train.model)

    generate_images(model, 100, vis_path)