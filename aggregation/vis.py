from turtle import color
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
from cmath import nan
from matplotlib import colors

'''
    Generate Visualizations

    TODO:    -generate batch-wise visualizations
            -move arbitrary paths to config.


    set a config setup for saving models, saving results, saving visualizations, 
    lightning_logs?, be able to tensorboard all of them?
'''

def generate_images(model, num_images, dir_path):
    train_loader, val_loader, test_loader = util.make_loaders(batch_size = 1, mode = 'vis', sample_mode='')

    c = 0
    with torch.no_grad():
        for sample in test_loader:
            image, masks, value = sample

            if cfg.train.model == 'gauss' or 'rsample':
                vals, vars = model.get_valOut(image)
            else:
                vals= model.get_valOut(image)   
                vars=0 # NO STANDARD DEVIATIONS

            estimated_values = model.pred_Out(image, masks)
            #estimated_values = estimated_values.cpu().detach().numpy()

            masks = masks[0]
            value = value[0]
            estimated_values= estimated_values[0]
            errors = mask_sum = pred_map = region_map = true_map = np.zeros_like(masks[0])

            for i,mask in enumerate(masks):
                mask = np.array(mask)
                region_map = np.add(mask*np.random.randint(0,1000), region_map)

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

            for i,mask in enumerate(masks):
                mask = np.array(mask)
                mask_sum += mask
                pixel_count = (mask == 1).sum()
                uniform_value = torch.abs((estimated_values[i]-value[i]))/pixel_count
                uniform_value = uniform_value.cpu().detach().numpy()
                errors = np.add(mask*uniform_value, errors)
            
            errors = errors.reshape(cfg.data.cutout_size)

            for i,mask in enumerate(masks):
                mask = np.array(mask)
                pixel_count = (mask == 1).sum()
                uniform_value = value[i]/pixel_count
                uniform_value = uniform_value.cpu().detach().numpy()
                true_map = np.add(mask*uniform_value, true_map)
            
            true_map = true_map.reshape(cfg.data.cutout_size)

            #stds = stds.squeeze(0).squeeze(0).cpu().detach().numpy() * mask_sum.reshape(512,512)
            #vals = vals.squeeze(0).squeeze(0).cpu().detach().numpy() * mask_sum.reshape(512,512)

            path =  os.path.join(dir_path, str(c) )

            if not(os.path.exists(path)):
                os.mkdir(path)

            generate_plot(image.squeeze(0),vals.squeeze(0),vars.squeeze(0),pred_map,true_map, region_map,errors, path)
            c+=1
            if c >= num_images:
                return

def generate_scatter(model, dir_path):

    est_pth = os.path.join(dir_path, 'estimated.pkl')
    val_pth = os.path.join(dir_path, 'value.pkl')
    scatter_pth = os.path.join(dir_path, 'scatter.png')
    density_pth = os.path.join(dir_path, 'density.png')
    mae_error_pth = os.path.join(dir_path, 'mae_error_hist.png')
    mse_error_pth = os.path.join(dir_path, 'mse_error_hist.png')

    with open(est_pth, "rb") as fp:
        estimated_arr = pickle.load(fp)
    with open(val_pth, "rb") as fp:
        value_arr = pickle.load(fp)

    plt.scatter(value_arr,estimated_arr, s=0.2)
    plt.title("Prediction Scatter Plot")
    plt.xlabel("True Value")
    plt.ylabel("Estimated Value")
    plt.savefig(scatter_pth)
    plt.close()

    plt.hist(value_arr, bins= 100)
    plt.title("Test-set Distribution")
    plt.xlabel("True Value")
    plt.savefig(os.path.join(dir_path,"true_value_hist.png"))
    plt.close()

    ax = sns.kdeplot(x = value_arr, y = estimated_arr,
     fill= True, thresh=0, levels =100,cmap="mako")
    ax.set(xlabel = "True Values", ylabel = "Estimated Value", title= "Prediction Density Plot")
    plt.savefig(density_pth)
    plt.close()

    mae_errors = np.abs( np.array(value_arr) - np.array(estimated_arr))
    print("MAE:")
    print("STD:",mae_errors.std(), "MEAN:", mae_errors.mean())
    plt.hist(mae_errors, bins = 1000)
    #plt.xlim(-2,2)
    plt.xlabel("Raw Error")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.savefig(mae_error_pth)
    plt.close()

    mse_errors = np.array(value_arr) - np.array(estimated_arr)
    mse_errors = np.power(mse_errors,2)
    print("MSE:")
    print("STD:",mse_errors.std(), "MEAN:", mse_errors.mean())
    plt.hist(mse_errors, bins = 1000)
    #plt.xlim(-2,2)
    plt.xlabel("Raw Error")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.savefig(mse_error_pth)
    plt.close()



def generate_plot(image,vals, vars, pred_map,true_map, region_map, error, path):

    min_lim = 0
    max_lim = 256

    #plt.set_title("Image")
    plt.imshow(image.permute(1,2,0)) 
    plt.tight_layout(pad=0)
    plt.axis('off')
    plt.xlim(min_lim, max_lim)
    plt.ylim(min_lim, max_lim)
    plt.savefig(os.path.join(path, "image"), bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.imshow(region_map , cmap = 'terrain')
    plt.tight_layout(pad=0)
    plt.axis('off')
    plt.xlim(min_lim, max_lim)
    plt.ylim(min_lim, max_lim)
    plt.savefig(os.path.join(path, "region_map"), bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.imshow(pred_map , cmap = 'Greens')
    plt.tight_layout(pad=0)
    plt.axis('off')
    plt.xlim(min_lim, max_lim)
    plt.ylim(min_lim, max_lim)
    plt.savefig(os.path.join(path, "pred_map"), bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.imshow(true_map , cmap = 'Greens')
    plt.tight_layout(pad=0)
    plt.axis('off')
    plt.xlim(min_lim, max_lim)
    plt.ylim(min_lim, max_lim)
    plt.savefig(os.path.join(path, "true_map"), bbox_inches='tight', pad_inches=0)
    plt.close()

    print(vals.shape)
    plt.imshow(vals.permute(1,2,0) , cmap = 'Greens')
    plt.tight_layout(pad=0)
    plt.axis('off')
    plt.xlim(min_lim, max_lim)
    plt.ylim(min_lim, max_lim)
    plt.savefig(os.path.join(path, "value_pred"), bbox_inches='tight', pad_inches=0)
    plt.close()
    #axs[0][1].set_title("Value Prediction")

    if cfg.train.model == 'gauss' or 'rsample':
        plt.imshow(vars.permute(1,2,0), cmap = 'Reds')
        #plt.tight_layout(pad=0)
        plt.axis('off')
        plt.colorbar()
        plt.xlim(min_lim, max_lim)
        plt.ylim(min_lim, max_lim)
        plt.savefig(os.path.join(path, "vars"))
        plt.close()

        plt.imshow(vars.permute(1,2,0), cmap = 'Reds', norm=colors.LogNorm())
        #plt.tight_layout(pad=0)
        plt.colorbar()
        plt.axis('off')
        plt.xlim(min_lim, max_lim)
        plt.ylim(min_lim, max_lim)
        plt.savefig(os.path.join(path, "vars_log"))
        plt.close()
        #axs[1][1].set_title("Variance")

        plt.imshow(vars.permute(1,2,0), cmap = 'Reds', vmin=0,vmax=1e-8)
        #plt.tight_layout(pad=0)
        plt.colorbar()
        plt.axis('off')
        plt.xlim(min_lim, max_lim)
        plt.ylim(min_lim, max_lim)
        plt.savefig(os.path.join(path, "vars_clipped"))
        plt.close()

        color_map = plt.cm.get_cmap('Blues')
        plt.imshow(vars.permute(1,2,0), cmap = color_map.reversed())
        #plt.tight_layout(pad=0)
        plt.colorbar()
        plt.axis('off')
        plt.xlim(min_lim, max_lim)
        plt.ylim(min_lim, max_lim)
        plt.savefig(os.path.join(path, "vars_reversed"))
        plt.close()

        color_map = plt.cm.get_cmap('Blues')
        plt.imshow(vars.permute(1,2,0), cmap = color_map.reversed() , vmin=0,vmax=1e-8)
        #plt.tight_layout(pad=0)
        plt.colorbar()
        plt.axis('off')
        plt.xlim(min_lim, max_lim)
        plt.ylim(min_lim, max_lim)
        plt.savefig(os.path.join(path, "vars_reversed_clipped"))
        plt.close()
        #axs[1][1].set_title("Variance")

        color_map = plt.cm.get_cmap('Blues')
        plt.imshow(vars.permute(1,2,0), cmap = color_map.reversed(), norm=colors.LogNorm())
        #plt.tight_layout(pad=0)
        plt.colorbar()
        plt.axis('off')
        plt.xlim(min_lim, max_lim)
        plt.ylim(min_lim, max_lim)
        plt.savefig(os.path.join(path, "vars_reversed_lognorm"))
        plt.close()
        #axs[1][1].set_title("Variance")

    plt.imshow(error, cmap = 'Greys')
    plt.tight_layout(pad=0)
    plt.axis('off')
    plt.xlim(min_lim, max_lim)
    plt.ylim(min_lim, max_lim)
    plt.savefig(os.path.join(path, "error_map"), bbox_inches='tight', pad_inches=0)
    plt.close()
    #axs[1][0].set_title("True Value Map")
  


if __name__ == '__main__':
    dir_path = os.path.join(os.getcwd(), 'results', cfg.experiment_name)
    vis_path= os.path.join(dir_path ,'visualizations2/')
    ckpt_path = os.path.join(dir_path,'best.ckpt')
    if not(os.path.exists(vis_path)):
        os.mkdir(vis_path)

    model = test.loadModel(ckpt_path , cfg.train.model)

    generate_scatter(model, dir_path)

    generate_images(model, 100, vis_path)
