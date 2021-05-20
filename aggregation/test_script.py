import trainAgg


# pick a sample image from the hennepin set

path = '\u\eag-d1\data\Hennepin\something'
# Use PIL
# turn it into a tensor
# give it a batch dimension

def test(dataloader):
    # initialize 4 of the region agg networks
    # they should have random initialization on the 4 random weights

    # next get predictions for one sample image, before REGION AGG
    # Need to create some 


    # train each for maybe 100 epochs 
    # save the weights?


    # then get value map for each on the sample image, should be converging to something.



if __name__ == '__main__':

    this_dataset = data_factory.dataset_hennepin('train','/u/eag-d1/data/Hennepin/ver7/',
    '/u/eag-d1/data/Hennepin/ver7/hennepin_bbox.csv',
    '/u/pop-d1/grad/cgar222/Projects/disaggregation/dataset/hennepin_county_parcels/hennepin_county_parcels.shp')

    torch.manual_seed(0)

    train_size = int( np.floor( len(this_dataset) * (1.0-cfg.train.validation_split-cfg.train.test_split) ) )
    val_size = int( np.ceil( len(this_dataset) * cfg.train.validation_split ))
    test_size = int(np.ceil( len(this_dataset) * cfg.train.test_split ))

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(this_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=cfg.train.shuffle, collate_fn = my_collate,
                             num_workers=cfg.train.num_workers)

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=cfg.train.shuffle, collate_fn = my_collate,
                             num_workers=cfg.train.num_workers)


    test(train_dataset)