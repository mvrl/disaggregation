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
'''
    Generate Visualizations

    TBD:    -generate batch-wise visualizations
            -move arbitrary paths to config.


    set a config setup for saving models, saving results, saving visualizations, 
    lightning_logs?, be able to tensorboard all of them?
'''

# This is a py script of the original jupyter notebook
# It will save all the generated visualizations to /outputs
# Come up with some naming scheme

def generate_images(model, num_images, dir_path):
    train_loader, val_loader, test_loader = util.make_vis_loaders(batch_size = 1, mode = 'vis', sample_mode='')

    c = 0
    with torch.no_grad():
        for sample in test_loader:
            image, masks, value, polygons, img_bbox = sample

            value = value[0]

            vals = model.get_valOut(image)

            estimated_values = model.pred_Out(image, masks)
            #estimated_values = estimated_values.cpu().detach().numpy()
            # CRATING AVERAGE VALUE MASK COMBINATION
            masks = masks[0]
            estimated_values = estimated_values[0]
            pred_map = np.zeros_like(masks[0])
            for i,mask in enumerate(masks):
                mask = np.array(mask)
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

            path =  os.path.join(dir_path, str(c) )

            generate_plot(image.squeeze(0),vals.squeeze(0),pred_map,true_map, path)
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
    plt.ylim(0,1000)
    plt.xlim(0,1000)
    plt.xlabel("True Value")
    plt.ylabel("Estimated Value")
    plt.savefig(scatter_pth)
    plt.close()

    plt.hist(value_arr, bins= 100)
    plt.title("Test-set Distribution")
    plt.xlabel("True Value")
    plt.savefig(os.path.join(dir_path,"true_value_hist.png"))
    plt.close()

    ax = sns.kdeplot(x = value_arr, y = estimated_arr, clip = (0,1000),
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



def generate_plot(image,vals, pred_map, true_map, path):
    fig, axs = plt.subplots(2,2,figsize=(10,10))
    axs[0][0].imshow(image.permute(1,2,0) )
    axs[0][0].axis('off')
    axs[0][0].set_title("Image")
    print(vals.shape)
    axs[0][1].imshow(vals.permute(1,2,0) , cmap = 'Greens')
    axs[0][1].axis('off')
    axs[0][1].set_title("Value Prediction")
    
    # CRATING AVERAGE VALUE MASK COMBINATION
    axs[1][1].imshow(pred_map, cmap = 'Greens')
    axs[1][1].axis('off')
    axs[1][1].set_title("Aggregated Value Prediction")

    # This needs proper color scalings... unsure how to do this
    axs[1][0].imshow(true_map, cmap = 'Greens')
    axs[1][0].set_title("True Value Map")
    axs[1][0].axis('off')
    fig.tight_layout(pad=1)
    plt.savefig(path)
    plt.close(fig)



if __name__ == '__main__':
    dir_path = os.path.join(os.getcwd(), 'results', cfg.experiment_name)
    vis_path= os.path.join(dir_path ,'visualizations/')
    ckpt_path = os.path.join(dir_path,'best.ckpt')
    if not(os.path.exists(vis_path)):
        os.mkdir(vis_path)

    model = test.loadModel(ckpt_path , cfg.train.model)

    generate_scatter(model, dir_path)

    generate_images(model, 100, vis_path)
