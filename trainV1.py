import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import dataloader
from torch.utils.data import Dataset, DataLoader
import models
import torchvision.transforms as transforms
import torch.cuda
import datetime
import torchvision
import matplotlib.pyplot as plt


transformation = transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor()])


def imshow(inp,title = None):
    inp = torchvision.utils.make_grid(inp)
    inp = inp.cpu().detach().numpy().transpose((1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    inp = std*inp[:,:,0:3] + mean
    inp = np.clip(inp,0,1)
    plt.imshow(inp)
    plt.pause(0.001)

def adjust_lr(optimizer,cur_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr/10

def get_loss(out1,heatmap,out2,pof,loss_func):
    loss1 = loss_func(out1,heatmap)
    loss2 = loss_func(out2,pof)
    total_loss = loss1+loss2
    return total_loss

def train(epochs = 80,batch = 8,device = torch.device('cpu')):
    train_data = dataloader.PairedDataset(mode = "train",transform = transformation)
    print(train_data.__len__())
    train_loader = DataLoader(dataset = train_data,batch_size = batch)
    
    #this part is val loader
    #val_data = dataloader.PairedDataset(mode = "val",transform = transformation)
    #val_loader = DataLoader(dataset = val_data,batch_size = batch)
    #this is the dataloader
    
    model = models.bodypose().to(device)
    #print(model)
    
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.MSELoss().to(device)
    #here we should set the loss and learning rate
    last_loss = 0
    current_lr = 0.001
    for epoch in range(epochs):
        model.train()
        print('epoch {}'.format(epoch+1))
        train_loss = 0
        
        for count,(images,batch_y,heatmap,pof)in enumerate(train_loader):
            optimizer.zero_grad()
            if -1 in batch_y:
                continue
            input_image = images.to(device)
            out1,out2 = model(input_image)
            heatmap = heatmap.to(device).float()
            pof = pof.to(device).float()
            loss = get_loss(out1,heatmap,out2,pof,loss_func)
            train_loss += loss.data
            loss.backward()
            optimizer.step()
            if count%4000 == 0:
                print(datetime.datetime.now())
                print(loss.data)
                #imshow(out1[0])
                #imshow(out2[0])
        loss_this_time =train_loss / (len(train_data))
        print('Train Loss: {:.6f}'.format(loss_this_time))
        if loss_this_time >= last_loss:
            adjust_lr(optimizer,current_lr)
            current_lr = current_lr/10
        last_loss = loss_this_time
        #here will be validatin
        """
        model.eval()
        eval_loss = 0
        for count,(images,batch_y,heatmap,pof)in enumerate(val_loader):
            if -1 in batch_y:
                continue
            input_image = images.to(device)
            out1,out2 = model(input_image)
            heatmap = heatmap.to(device).float()
            pof = pof.to(device).float()
            loss = get_loss(out1,heatmap,out2,pof,loss_func)
            eval_loss += loss.data
        loss_vali =eval_loss / (len(val_data))
        print('Val Loss: {:.6f}'.format(loss_vali))
        """
        if epoch%3 == 0:
            torch.save(model,'multiepoch'+str(epoch)+'.pkl')
    torch.save(model,'multiresult.pkl')

device_2 = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
train(80,1,device=device_2)
