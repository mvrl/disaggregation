import matplotlib.pyplot as plt
import util
import trainAgg
import data_factory
import torch
import os
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import pickle
import seaborn as sns
from config import cfg
'''
    Generate Visualizations

    TBD:    -generate batch-wise visualizations
            -move arbitrary paths to config.
'''

# This is a py script of the original jupyter notebook
# It will save all the generated visualizations to /outputs
# Come up with some naming scheme

def generate_images(experiment_name, model, num_images):
    train_loader, val_loader, test_loader = util.make_loaders(batch_size = 1, mode = 'test')

    #test_loader.batch_size = 1
    #this_dataset.mode = 'test'
    #print(test_loader)

    dir_path= os.path.join( os.getcwd(),'visualizations/',experiment_name )
    os.makedirs(dir_path, exist_ok=True)
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
            uniform_value_map = np.zeros_like(masks[0])
            for i,mask in enumerate(masks):
                mask = np.array(mask)
                pixel_count = (mask == 1).sum()
                uniform_value = estimated_values[i]/pixel_count
                uniform_value = uniform_value.cpu().detach().numpy()
                uniform_value_map = np.add(mask*uniform_value, uniform_value_map)
            

            uniform_value_map = uniform_value_map.reshape(cfg.data.cutout_size)

            path =  os.path.join(dir_path, str(c) )
            print(path)
            print(c)

            generate_plot(image.squeeze(0),vals.squeeze(0),uniform_value_map,polygons,img_bbox, path)
            c+=1
            if c >= num_images:
                return

def generate_pred_lists(model, dir_path):
    train_loader, val_loader, test_loader = util.make_loaders(batch_size = 1, mode = 'test')

    est_pth = os.path.join(dir_path, 'estimated.txt')
    val_pth = os.path.join(dir_path, 'value.txt')
    
    estimated_arr = []
    value_arr = []

    print("Computing all predicted values...")
    with torch.no_grad():
        for sample in tqdm(test_loader):
            image, mask, value = sample

            estimated_values = model.pred_Out(image, mask)

            estimated_arr.extend( estimated_values[0].cpu().numpy().tolist())
            value_arr.extend(value[0].numpy().tolist())

    with open(est_pth, "wb") as fp:   #Pickling
        pickle.dump(estimated_arr, fp)
    with open(val_pth, "wb") as fp:   #Pickling
        pickle.dump(value_arr, fp)

    return estimated_arr, value_arr

def generate_scatter(experiment_name, model):
    dir_path= os.path.join( os.getcwd(),'visualizations/',experiment_name)
    if not(os.path.exists(dir_path)):
        os.mkdir(dir_path)

    est_pth = os.path.join(dir_path, 'estimated.txt')
    val_pth = os.path.join(dir_path, 'value.txt')
    scatter_pth = os.path.join(dir_path, 'scatter.png')
    density_pth = os.path.join(dir_path, 'density.png')
    mae_error_pth = os.path.join(dir_path, 'mae_error_hist.png')
    mse_error_pth = os.path.join(dir_path, 'mse_error_hist.png')

    if not(os.path.exists(est_pth)):
        estimated_arr, value_arr = generate_pred_lists(model, dir_path)
    else:
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



def generate_plot(image,vals, uniform_value_map, polygons, img_bbox, path):
    #crop = transforms.CenterCrop((296,296))

    fig, axs = plt.subplots(4,1,figsize=(10,15))
    axs[0].imshow(image.permute(1,2,0) )
    axs[0].axis('off')
    axs[0].set_title("Image")
    axs[1].imshow(vals.permute(1,2,0) , cmap = 'Greens')
    axs[1].axis('off')
    axs[1].set_title("Value Map")
    
    # CRATING AVERAGE VALUE MASK COMBINATION
    axs[2].imshow(uniform_value_map, cmap = 'Greens')
    axs[2].axis('off')
    axs[2].set_title("Value by Parcel Sum")

    # This needs proper color scalings... unsure how to do this
    polygons[0].plot(ax=axs[3], column = 'AVERAGE_MV1', alpha = 0.95, linewidth=5, cmap = 'Greens')
    axs[3].imshow(image.squeeze(0).permute(1,2,0), extent = img_bbox[0], origin = 'upper', alpha= 0.3)
    axs[3].set_title("True Value Map")
    axs[3].axis('off')
    fig.tight_layout()
    plt.savefig(path)
    plt.close(fig)

# Thinking about plotting the masks potentially. 
def plot_masks(image, masks):
    fig,axs = plt.subplots(1,len(masks)+1, figsize = (15,5))
    axs[0].imshow(image.squeeze(0).permute(1,2,0))
    axs[0].axis('off')
    axs[0].set_title("Image")

    for x in range(1, len(masks)+1):
        axs[x].plot()



def end2end_model():
    model = trainAgg.RALModule(use_pretrained=False)
    #'/u/pop-d1/grad/cgar222/Projects/disaggregation/aggregation/lightning_logs/version_215/checkpoints/epoch=124-step=25999.ckpt' NEWEST
    # september '/u/pop-d1/grad/cgar222/Projects/disaggregation/aggregation/lightning_logs/version_206/checkpoints/epoch=196-step=44521.ckpt'
    # OCTOBER
    #model = model.load_from_checkpoint('/u/pop-d1/grad/cgar222/Projects/disaggregation/aggregation/lightning_logs/version_215/checkpoints/epoch=124-step=25999.ckpt', 
    #    use_pretrained=False,use_existing=True)
    model = model.load_from_checkpoint('/u/pop-d1/grad/cgar222/Projects/disaggregation/aggregation/lightning_logs/version_276/checkpoints/epoch=120-val_loss=3327.04-train_loss=2201.34.ckpt', 
        use_pretrained=False)
    return model

def onexone_model():
    model = trainAgg.OnexOneAggregationModule(use_pretrained=False)
    #model = model.load_from_checkpoint('/u/pop-d1/grad/cgar222/Projects/disaggregation/aggregation/lightning_logs/version_218/checkpoints/epoch=101-step=21215.ckpt', 
    #    use_pretrained=False, use_existing=True)
    model = model.load_from_checkpoint('/u/pop-d1/grad/cgar222/Projects/disaggregation/aggregation/lightning_logs/version_251/checkpoints/epoch=164-step=17159.ckpt', 
        use_pretrained=False)
    return model

def uniform_model():
    model = trainAgg.UniformModule(use_pretrained=False)
    model = model.load_from_checkpoint('/u/pop-d1/grad/cgar222/Projects/disaggregation/aggregation/lightning_logs/version_277/checkpoints/epoch=214-val_loss=0.04-train_loss=0.01.ckpt', 
        use_pretrained=False)
    return model

if __name__ == '__main__':

    model = end2end_model()
    #model = onexone_model()
    #model = uniform_model()

    experiment_name = 'october_vis_agg'

    generate_images(experiment_name, model, 100)

    #generate_scatter(experiment_name, model)