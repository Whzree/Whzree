#!/usr/bin/env python
# -*- coding:UTF-8 -*-
'''
@Project:pythonProject1
@File:tensor4.py
@IDE:pythonProject1
@Author:whz
@Date:2025/3/17 12:14

'''
from random import shuffle

import torch
from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim


"""
data_path = "./img/data"
cifar10 = datasets.CIFAR10(data_path,train=True,download=True)
cifar10_val = datasets.CIFAR10(data_path,train=False,download=True)
print(len(cifar10))
img,label = cifar10[99]
#plt.imshow(img)
#plt.show()
#将数据集转换为张量
to_tensor = transforms.ToTensor()
img_t = to_tensor(img)
tensor_cifar10 = datasets.CIFAR10(data_path,train=True,download=False,transform=transforms.ToTensor())
img_t,a=tensor_cifar10[99]
#a 是类别
#print(img_t)
#print(a)
#数据归一化
imgs = torch.stack([img_t for img_t,_ in tensor_cifar10],dim=3)
print(imgs.shape)
#torch.Size([3, 32, 32, 50000]) C,W,H,N
imgs.view(3,-1).mean(dim=1)
imgs.view(3,-1).std(dim=1)
"""
#区分鸟和飞机

#构建数据集
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


def Softmax(x):
    return torch.exp(x) / torch.exp(x).sum()

#分类数
n_out = 2
#模型定义 输入特征数 32 * 32 * 7
model = nn.Sequential(
    nn.Linear(3072,1024),
    nn.Tanh(),
    nn.Linear(1024, 512),
    nn.Tanh(),
    nn.Linear(512, 128),
    nn.Tanh(),
    nn.Linear(128,n_out),
    nn.LogSoftmax(dim=1)
)

def softmax(x):
    return torch.exp(x) / torch.exp(x).sum()

# 输入为 3072 标量
img,_ = cifar2[0]

img_batch = img.view(-1).unsqueeze(0)
out = model(img_batch)
#torch.max 返回最大值以及其对应的索引

learning_rate = 1e-2
optimizer = optim.SGD(model.parameters(),lr=learning_rate)

#第一个参数接受LogSoftmax的输出,第二个参数
loss_fn = nn.NLLLoss()

n_epochs = 100
"""
for epoch in range(n_epochs):
    for img,label in cifar2:
        out = model(img.view(-1).unsqueeze(0))
        loss = loss_fn(out,torch.tensor([label]))

        #清楚导数缓存垃圾值
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    print("Epoch: %d,Loss:%f"%(epoch,float(loss)))
"""



#在小批量上平均更新
train_loader = torch.utils.data.DataLoader(cifar2,batch_size=64,shuffle=True)


#在训练代码中,我们选择大小为 1 的小批量
for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        batch_size = imgs.shape[0]
        # batch_size = 64
        outputs = model(imgs.view(batch_size,-1))
        loss = loss_fn(outputs,labels)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


    print("Epoch: %d,Loss:%f" % (epoch, float(loss)))

#测试验证集
val_loader = torch.utils.data.DataLoader(cifar2_val,batch_size=64,shuffle=False)
correct = 0
total = 0
with torch.no_grad():
    for imgs,labels in val_loader:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size,-1))
        _,predicted = torch.max(outputs,dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
print("Accuracy : %f",correct / total)

