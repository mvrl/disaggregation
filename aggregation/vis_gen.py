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
    os.mkdir(dir_path)
    i = 0
    with torch.no_grad():
        for sample in test_loader:
            image, mask, value, polygons, img_bbox = sample

            vals = model.get_valOut(image)

            path =  os.path.join(dir_path, str(i) )

            generate_plot(image,vals,polygons,img_bbox, path)
            i+=1
            if i >= num_images:
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
            image, mask, value, polygons, img_bbox = sample

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
    error_pth = os.path.join(dir_path, 'error_hist.png')

    if not(os.path.exists(est_pth)):
        estimated_arr, value_arr = generate_pred_lists(model, dir_path)
    else:
        with open(est_pth, "rb") as fp:
            estimated_arr = pickle.load(fp)
        with open(val_pth, "rb") as fp:
            value_arr = pickle.load(fp)

    plt.scatter(value_arr,estimated_arr, s=0.2)
    plt.title("Prediction Scatter Plot")
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.xlabel("True Value")
    plt.ylabel("Estimated Value")
    plt.savefig(scatter_pth)
    plt.close()

    plt.hist(value_arr, bins= 100)
    plt.title("Test-set Distribution")
    plt.xlabel("True Value")
    plt.savefig(os.path.join(dir_path,"true_value_hist.png"))
    plt.close()

    ax = sns.kdeplot(x = value_arr, y = estimated_arr, clip = (0,1),
     fill= True, thresh=0, levels =100,cmap="mako")
    ax.set(xlabel = "True Values", ylabel = "Estimated Value", title= "Prediction Density Plot")
    plt.savefig(density_pth)
    plt.close()

    errors = np.array(value_arr) - np.array(estimated_arr)
    print("STD:",errors.std(), "MEAN:", errors.mean())
    plt.hist(errors, bins = 1000)
    plt.xlim(-2,2)
    plt.xlabel("Raw Error")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.savefig(error_pth)
    plt.close()



def generate_plot(image,vals, polygons, img_bbox, path):
    crop = transforms.CenterCrop((296,296))

    fig, axs = plt.subplots(3,1,figsize=(10,15))
    axs[0].imshow(crop( image.squeeze(0)).permute(1,2,0) )
    axs[0].axis('off')
    axs[0].set_title("Image")
    axs[1].imshow(crop( vals.squeeze(0) ).permute(1,2,0) , cmap = 'Greens')
    print(torch.max(vals))
    axs[1].axis('off')
    axs[1].set_title("Value Map")
    # This needs proper color scalings... unsure how to do this
    polygons[0].plot(ax=axs[2], column = 'AVERAGE_MV1', alpha = 0.95, linewidth=5, cmap = 'Greens', vmin = 0, vmax = 1)
    axs[2].imshow(image.squeeze(0).permute(1,2,0), extent = img_bbox[0], origin = 'upper', alpha= 0.6)

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
    model = trainAgg.End2EndAggregationModule(use_pretrained=False, use_existing=True)
    #'/u/pop-d1/grad/cgar222/Projects/disaggregation/aggregation/lightning_logs/version_215/checkpoints/epoch=124-step=25999.ckpt' NEWEST
    # september '/u/pop-d1/grad/cgar222/Projects/disaggregation/aggregation/lightning_logs/version_206/checkpoints/epoch=196-step=44521.ckpt'
    model = model.load_from_checkpoint('/u/pop-d1/grad/cgar222/Projects/disaggregation/aggregation/lightning_logs/version_215/checkpoints/epoch=124-step=25999.ckpt', 
        use_pretrained=False,use_existing=True)
    return model

def onexone_model():
    model = trainAgg.OnexOneAggregationModule(use_pretrained=False, use_existing=True)
    model = model.load_from_checkpoint('/u/pop-d1/grad/cgar222/Projects/disaggregation/aggregation/lightning_logs/version_218/checkpoints/epoch=101-step=21215.ckpt', 
        use_pretrained=False, use_existing=True)
    return model

if __name__ == '__main__':

    model = end2end_model()
    #model = onexone_model()

    experiment_name = 'testing_scatter'

    #generate_images('end2end_testing_noNORM', model, 100)

    generate_scatter(experiment_name, model)
    


    