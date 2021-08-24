import matplotlib.pyplot as plt
import util
import trainAgg
import data_factory
import torch
import os

'''
    Generate Visualizations

    TBD:    -generate batch-wise visualizations
            -move arbitrary paths to config.
'''

# This is a py script of the original jupyter notebook
# It will save all the generated visualizations to /outputs
# Come up with some naming scheme

def generate_images(experiment_name, model, num_images):
    this_dataset = data_factory.dataset_hennepin('test','/u/eag-d1/data/Hennepin/ver8/',
        '/u/eag-d1/data/Hennepin/ver8/hennepin_bbox.csv',
        '/u/eag-d1/data/Hennepin/hennepin_county_parcels/hennepin_county_parcels.shp')

    dir_path= os.path.join( os.getcwd(),'visualizations/',experiment_name )
    os.mkdir(dir_path)
    i = 0
    with torch.no_grad():
        for sample in this_dataset:
            image = sample['image']
            polygons = sample['polygons']
            img_bbox = sample['img_bbox']

            vals = model.get_valOut(torch.unsqueeze(image,0))

            path =  os.path.join(dir_path, str(i) )

            generate_plot(image,vals,polygons,img_bbox, path)
            i+=1
            if i >= num_images:
                return

def generate_plot(image,vals, polygons, img_bbox, path):
    fig, axs = plt.subplots(3,1,figsize=(10,15))
    axs[0].imshow(image.permute(1,2,0))
    axs[0].axis('off')
    axs[0].set_title("Image")
    axs[1].imshow(vals.squeeze(0).permute(1,2,0), vmin = 0.0, cmap = 'Greens')
    axs[1].axis('off')
    axs[1].set_title("Value Map")
    # This needs proper color scalings... unsure how to do this
    polygons.plot(ax=axs[2], column = 'AVERAGE_MV1', alpha = 0.95, linewidth=5, cmap = 'Greens', vmin = 0, vmax =1)
    axs[2].imshow(image.permute(1,2,0), extent = img_bbox, origin = 'upper', alpha= 0.6)

    fig.tight_layout()
    plt.savefig(path)
    plt.close(fig)

def end2end_model():
    model = trainAgg.End2EndAggregationModule(use_pretrained=False)
    model = model.load_from_checkpoint('/u/eag-d1/data/Hennepin/model_checkpoints/fixed.ckpt', 
        use_pretrained=False)
    return model

def onexone_model():
    model = trainAgg.aggregationModule(use_pretrained=False)
    model = model.load_from_checkpoint('/u/eag-d1/data/Hennepin/model_checkpoints/UNet_pretrained1x1.ckpt', 
        use_pretrained=False)
    return model

if __name__ == '__main__':

    model = end2end_model()
    #model = onexone_model()

    generate_images('end2end_test_fixed', model, 100)
    


    