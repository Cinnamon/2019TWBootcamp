
# coding: utf-8

# In[2]:


import torch
import torchvision.transforms as transforms

import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable

import time
from loss import MultiBoxLoss_2


# In[3]:


from Datagen import ListDataset
from ssd import SSD
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(train_file, test_file, num_epoch):
    use_gpu = torch.cuda.is_available()
    Loss = MultiBoxLoss_2()                      ## loss
    learning_rate = 0.01
    num_epochs = num_epoch
    batch_size = 4
    model =  SSD(depth=50, width=1)
    #optimizer = torch.optim.SGD([{"params":model.parameters()}], lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam([{"params":model.parameters()}], lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer)
    
    if use_gpu:
        model.cuda()
    
    model.train()

    train_dataset = ListDataset(root='GUN/WeaponS/',list_file=train_file, train =True, transform =transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle=True, num_workers=2)
    test_dataset = ListDataset(root='GUN/WeaponS/',list_file=test_file, train =True, transform =transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size = 16, shuffle=True, num_workers=2)


    for epoch in range(num_epochs):
        t1 = time.time()
        model.train()
        
        total_loss, valid_loss = 0,0 
        
        # Adjust learninig rate
        
        
        ## train model
        print("Train {} epoch: ".format(epoch+1))
        for i,(imgs, loc, conf) in enumerate(train_loader):
            imgs,loc, conf = Variable(imgs),Variable(loc),Variable(conf)
            if use_gpu:
                imgs = imgs.cuda()
                loc = loc.cuda()
                conf = conf.cuda()
            loc_pred, con_pred = model(imgs)
            
            loss = Loss(loc_pred, loc, con_pred, conf)
            total_loss += loss.item()
            #loss = conf_loss + loc_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #print('Training progress %.1f %%' %(100*(i+1)/len(train_loader)), end='')
        #print('loc loss: ', loc_loss_total/len(train_loader))
        #print('conf loss: ', conf_loss_total/len(train_loader))
        print('\rEpoch [%d/%d], Training loss: %.4f'
            % (epoch + 1, num_epochs, total_loss/len(train_loader)),end='\n')

        ## test model
        
        model.eval()
        with torch.no_grad():
            for i,(imgs, loc, conf) in enumerate(test_loader):
                imgs,loc, conf = Variable(imgs),Variable(loc),Variable(conf)
                if use_gpu:
                    imgs = imgs.cuda()
                    loc = loc.cuda()
                    conf = conf.cuda()
                loc_pred, con_pred = model(imgs)
                loss = Loss(loc_pred, loc, con_pred, conf)
                valid_loss += loss.item()
            
                #print('Validing progress %.1f %%' %(100*(i+1)/len(test_loader)), end='')
            print('\rEpoch [%d/%d], Validing loss: %.4f'
                   % (epoch + 1, num_epochs, valid_loss/len(test_loader)),end='\n')
            print('\n')
        scheduler.step(valid_loss)
        
        t2 = time.time()
        #print('epoch escape time %f secs' %t2-t1)
        
        # Save model
        #PATH_1 = 'drive/My Drive/BootCamp4/SSD/ssd_2.pki'
        #torch.save(model, PATH_1)
        
        PATH = 'drive/My Drive/BootCamp4/SSD/ssd_state_dict.pki'
        torch.save(model.state_dict(), PATH)
        
        
        


def main():
    train_file = 'yolo_train.txt'
    test_file = 'yolo_test.txt'
    train(train_file, test_file, 25)                                                                                                                            


# In[5]:


get_ipython().system('jupyter nbconvert --to script train.ipynb')

