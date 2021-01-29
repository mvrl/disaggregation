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

torch.backends.cudnn.benchmark = True

model = get_network(cfg.model.name)

# dataloaders
train_loader = get_data(cfg, mode='train')
val_loader = get_data(cfg, mode='val')

# Optimizer
optim = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.l2_reg)

# lr schedular
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=cfg.train.lr_decay_every, gamma=cfg.train.lr_decay)

# loss function
weight_list = cfg.train.loss_weight
if len(weight_list)==0:
    seg_criterion = nn.CrossEntropyLoss(ignore_index=-1) 
else:
    weight = torch.tensor(weight_list).cuda()
    print('class weights for loss:', weight)
    seg_criterion = nn.CrossEntropyLoss(weight, ignore_index=-1)

reg_criterion = nn.SmoothL1Loss()

# logging
train_loss = []
test_loss = []
best_val_loss = 999.0

out_dir = cfg.train.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
else:
    print('output directory ', out_dir, ' already exists. Make sure you are not overwriting previously trained model...')

print('configurations: ', cfg)
print('starting training')


## Train
for epoch in range(cfg.train.num_epochs):
    # training
    loss_train = 0
    model.train()
    for i, data in enumerate(train_loader):
        optim.zero_grad()  # clear gradients        
        image = data[0].cuda()
        seg_labels = data[1].long().cuda()
        reg_labels = data[2].float().cuda()
        
        seg_predictions, reg_predictions = model(image)
        
        seg_loss = seg_criterion(seg_predictions, seg_labels).unsqueeze(0)
        
        #seg_loss.backward()
    
        #optim.step()

        #optim.zero_grad()hi 

        #print( list(reg_labels.size() ) )
        #print( list(seg_labels.size() ) )
        #print( list(reg_predictions.size() ) )
        #print( list(seg_predictions.size() ) )

        reg_predictions = torch.squeeze(reg_predictions)

        reg_loss = reg_criterion( reg_predictions , reg_labels)
        #reg_loss.backward()

        loss = reg_loss + seg_loss

        loss.backward()

        optim.step()
        
        loss_train += seg_loss.detach().cpu().item() #+ reg_loss.detach().cpu().item()
        
        # printing
        if (i+1)%20 == 0:
            print('[Ep ', epoch+1, (i+1), ' of ', len(train_loader) ,'] train loss: ', loss_train/(i+1))
            
        
    # end of training loop
    loss_train /= len(train_loader)
    
    loss_val = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            optim.zero_grad()  # clear gradients
            image = data[0].cuda()
            seg_labels = data[1].long().cuda()
            reg_labels = data[2].float().cuda()
        
            seg_predictions, reg_predictions = model(image)
            seg_loss = seg_criterion(seg_predictions, seg_labels).unsqueeze(0)

            reg_predictions = torch.squeeze(reg_predictions)
            reg_loss = reg_criterion( reg_predictions , reg_labels)

            loss = reg_loss + seg_loss

            loss_val += loss.detach().cpu().item()
                
    # end of validation
    loss_val /= len(val_loader)
    
    # End of epoch
    scheduler.step()
    
    train_loss.append(loss_train)
    test_loss.append(loss_val)
    
    print('End of epoch ', epoch+1, ' , Train loss: ', loss_train, ', val loss: ', loss_val)   
    
    # save best model checkpoint
    if loss_val < best_val_loss:
        best_val_loss = loss_val
        fname = 'model_dict.pth'
        torch.save(model.state_dict(), os.path.join(out_dir, fname))
        print('=========== model saved at epoch: ', epoch+1, ' =================')

        
# save model checkpoint at the end
fname = 'model_dict_end.pth'
torch.save(model.state_dict(), os.path.join(out_dir, fname))
print('model saved at the end of training: ')        

# save loss curves        
plt.figure()
plt.plot(train_loss)
plt.plot(test_loss)
plt.legend(['train loss', 'test loss'])
fname = os.path.join(out_dir,'loss.png')
plt.savefig(fname)
# plt.show()

# Saving logs
log_name = os.path.join(out_dir, "logging.txt")
with open(log_name, 'w') as result_file:
    result_file.write('Logging... \n')
    result_file.write('Validation loss ')
    result_file.write(str(test_loss))
    result_file.write('\nTraining loss  ')
    result_file.write(str(train_loss))

