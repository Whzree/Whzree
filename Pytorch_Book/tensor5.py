#!/usr/bin/env python
# -*- coding:UTF-8 -*-
'''
@Project:pythonProject1
@File:tensor6.py
@IDE:pythonProject1
@Author:whz
@Date:2025/3/17 17:47

'''
from random import shuffle

import torch

from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime


#数据准备
data_path = "./img/data"
cifar10 = datasets.CIFAR10(data_path,train=True,download=True,transform=transforms.ToTensor())
cifar10_val = datasets.CIFAR10(data_path,train=False,download=True,transform=transforms.ToTensor())
tensor_cifar10 = datasets.CIFAR10(data_path,train=True,download=False,transform=transforms.ToTensor())
label_map = {0:0,2:1}
class_names = ['airplane','bird']
cifar2 =[(img,label_map[label])
         for img,label in cifar10
         if label in [0,2]]
cifar2_val =[(img,label_map[label])
         for img,label in cifar10_val
         if label in [0,2]]

conv = nn.Conv2d(3,16,kernel_size=3)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(16,8,kernel_size=3,padding=1)
        self.fc1 = nn.Linear(8*8*8,32)
        self.fc2 = nn.Linear(32,2)

    def forward(self,x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)),2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)),2)
        out = out.view(-1,8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

model = Net()
def training_loop(n_epochs,optimizer,model,loss_fn,train_loader):
    for epoch in range(1,n_epochs+1):
        loss_train = 0.0
        for imgs , labels in train_loader:
            outputs = model(imgs)

            loss = loss_fn(outputs,labels)

            #L2正则化
            l2_lamdba = 0.001
            l2_norm = sum(p.pow(2.0).sum()
                          for p in model.parameters())
            loss = loss + l2_lamdba * l2_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0 :
            print("{}epoch{},Training loss{}".format(datetime.datetime.now(),epoch,loss_train/len(train_loader)))

train_loader = torch.utils.data.DataLoader(cifar2,batch_size=64,shuffle=True)
optimizer = optim.SGD(model.parameters(),lr=1e-2)
loss_fn = nn.CrossEntropyLoss()
model = Net()

training_loop(n_epochs=100,optimizer=optimizer,model=model,loss_fn=loss_fn,train_loader=train_loader)


#测量准确率
val_loader = torch.utils.data.DataLoader(cifar2_val,batch_size=64,shuffle=False)

def validate(model,train_loader,val_loader):
    for name ,loader in [("train",train_loader),("val",val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs , labels in loader:
                outputs = model(imgs)
                _,predicted = torch.max(outputs,dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
        print("Accuracy {}:{:.2f}".format(name,correct/total))
validate(model,train_loader,val_loader)



# Dropout 正则化
class NetDropout(nn.Module):
    def __init__(self,n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3,n_chans1,kernel_size=3,padding=1)
        self.conv1_dropout = nn.Dropout2d(p=0.4)
        self.conv2 = nn.Conv2d(n_chans1,n_chans1 // 2,kernel_size=3,padding=1)
        self.conv2_dropout = nn.Dropout2d(p=0.4)
        self.fc1 = nn.Linear(8 * 8 * n_chans1 //2 ,32)
        self.fc2 = nn.Linear(32,2)
    def forward(self,x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)),2)
        out = self.conv1_dropout(out)
        out = F.max_pool2d(torch.tanh(self.conv2(out)),2)
        out = self.conv2_dropout(out)
        out = out.view(-1,8 * 8 * self.n_chans1 // 2)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out



# 批量归一化

class NetBatchNorm(nn.Module):
    def __init__(self,n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3,n_chans1,kernel_size=3,padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_chans1)
        self.conv2 = nn.Conv2d(n_chans1,n_chans1 // 2,kernel_size=3,padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_chans1//2)
        self.fc1 = nn.Linear(8 * 8 * n_chans1 //2 ,32)
        self.fc2 = nn.Linear(32,2)
    def forward(self,x):
        out = self.conv1_batchnorm(self.conv1(x))
        out = F.max_pool2d(torch.tanh(out),2)
        out = self.conv2_batchnorm(self.conv2(out))
        out = F.max_pool2d(torch.tanh(out),2)
        out = out.view(-1,8 * 8 * self.n_chans1 // 2)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

#定义非常深的网络
class ResBlock(nn.Module):
    def __init__(self,n_chans1):
        super(ResBlock,self).__init__()
        self.conv = nn.Conv2d(n_chans1,n_chans1,kernel_size=3,padding=1,bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans1)
        #初始化
        torch.nn.init.kaiming_normal(self.conv.weight,nonlinearity = 'relu')
        torch.nn.init.constant_(self.conv.weight,0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)
    def forward(self,x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x

class NetResDeep(nn.Module):
    def __init__(self,n_chans1 = 32 , n_blocks = 100):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3,n_chans1,kernel_size=3,padding=1)

        self.resblocks = nn.Sequential(
            *(n_blocks * [ResBlock(n_chans1=n_chans1)])
        )

        self.fc1 = nn.Linear(8*8*n_chans1,32)
        self.fc2 = nn.Linear(32,2)

    def forward(self,x):
        out = F.max_pool2d(torch.relu(self.conv1(x)),2)

        out = self.resblocks(out)

        out=F.max_pool2d(out,2)
        out = out.view(-1,8*8*self.n_chans1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)

        return out












