from abc import abstractmethod
import trainAgg
import util

# pick a sample image from the hennepin set

path = '\u\eag-d1\data\Hennepin\something'
# Use PIL
# turn it into a tensor
# give it a batch dimension

def test(dataloader):
    # initialize 4 of the region agg networks/modules

    # they should have random initialization on the 4 random weights

    # next get predictions for one sample image, before REGION AGG
    # Need to create some 


    # train each for maybe 100 epochs 
    # save the weights?


    # then get value map for each on the sample image, should be converging to something.



if __name__ == '__main__':

    train_loader, val_loader, test_loader = util.make_loaders()

    test(train_loader)