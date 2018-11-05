from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
import scipy.misc as smi
import copy
from model import FCN8
from createDataset import MyDataset
import math
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
max_epoch = 30
labels = 40
dataroot = './data'
batch = 10
save_after = 1
lr = 1.0e-6
gpu = '0'
save_file = 'weights.pth'
img_size = (128,128)
momentum = 0.9

# class Test(nn.Module):
#     def __init__(self, weight=None, size_average=True, ignore_index=255):
#         super(Test, self).__init__()
#         self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

#     def forward(self, inputs, targets):
#         return self.nll_loss(F.log_softmax(inputs), targets)

def cross_entropy2d(img, target, weight=None, size_average=True):
    target = target.squeeze(1)
    n,c,h,w = img.size()
    # groundtruth = np.unique(target.cpu().numpy()[1,:,:])
    # print(groundtruth)
    log_p = F.log_softmax   (img, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduce=True)
    return loss

transform=transforms.Compose([transforms.Resize(img_size, interpolation=1),
    transforms.ToTensor()])

train_dataset = MyDataset(dataroot, transforms=transform, phase='train')
train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_dataset = MyDataset(dataroot, transforms=transform, phase='val')
val_dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch, shuffle=True)


# for i,data in enumerate(dataset_loader):
#         img = data[0].cuda()
#         seg = data[1].cuda()
#         print(img.shape)


# Doubt here
model = FCN8().to(device)
model.copy_parameters(model.loadVGG())

#  Get this cross checked

# criterion = Test(size_average=False).cuda()
optimizer = torch.optim.Adam([
    {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'],'lr': 2 * lr}, 
    {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'], 'lr': lr, 'weight_decay': 0.005}],
    betas=(momentum, 0.9))

f = open("losslog.txt", "w")

train_err = []
val_err = []
epoch = 0
while(epoch<max_epoch):
    for j in ['train', 'val']:
        dataset_loader = val_dataset_loader
        if( j == 'train'):
            dataset_loader = train_dataset_loader
        epoch_start = time.time()
        count=0
        flag = True
        running_loss = 0
        for i,data in enumerate(dataset_loader):
            img = data[0].to(device).float()
            seg = data[1].to(device).long()
            # seg = seg.unsqueeze(1)
            print(img.shape)
            print(seg.shape)            
            optimizer.zero_grad()
            out = model(img)
            # print(out.shape)
            loss = cross_entropy2d(out, seg, size_average=False)
            loss = loss / batch
            running_loss += loss.item()
            if(j == 'val'):
                lbl_pred = torch.max(out.data,1)[1].cpu().numpy()
                lbl_true = seg.data.cpu().numpy()
                plt.imshow(lbl_pred[0,:,:])
                if(flag):
                    plt.savefig(str(epoch)+'.png')
                    flag = False
                imgs = img.data.cpu()
                # for im, lt, lp in zip(imgs, lbl_true, lbl_pred):
                #     im, lt = dataset_loader.dataset.untransform(im, lt)
                #     label_trues.append(lt)
                #     label_preds.append(lp)
                continue
            print(j,i)
            loss.backward()
            optimizer.step()
            
        if(epoch%save_after==0):
            torch.save(model.state_dict(), save_file)
            # net.cuda(gpu)
        print(running_loss)
        epoch_loss = running_loss 

        if( j == 'val'):
            val_err.append(epoch_loss)
        else:
            train_err.append(epoch_loss)
        s = "Loss in phase "+j+" = "+str(epoch_loss)
        print(s)
        s = 'Time after completion of epoch '+str(epoch)+' = '+str(time.time()-epoch_start)
        print(s)
        f.write(s)
    epoch+=1
for i in range(len(val_err)):
    s = str(train_err[i])+" "+str(val_err[i])
    f.write(s)
f.close()








