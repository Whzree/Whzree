#!/usr/bin/env python
# -*- coding:UTF-8 -*-
'''
@Project:pythonProject1
@File:tensor2.py
@IDE:pythonProject1
@Author:whz
@Date:2025/3/15 21:32

'''
import numpy as np
import torch
import imageio.v2 as imageio
import os
import csv
from matplotlib import pyplot as plt



t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)
#定义模型
def model(t_u,w,b):
    return w * t_u + b
#定义损失函数
def loss_fn(t_p,t_c):
    squared_diffs = (t_p-t_c )**2
    return squared_diffs.mean()
w = torch.ones(())
b = torch.zeros(())
t_p = model(t_u,w,b)
loss = loss_fn(t_p,t_c)

#计算导数
def dloss_fn(t_p,t_c):
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)
    return dsq_diffs
def dmodel_dw(t_u,w,b):
    return t_u
def dmodel_db(t_u,w,b):
    return 1.0

#定义梯度函数
def grad_fn(t_c,t_u,t_p,w,b):
    dloss_dtp = dloss_fn(t_p,t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u,w,b)
    dloss_db = dloss_dtp * dmodel_db(t_u,w,b)
    return torch.stack([dloss_dw.sum(),dloss_db.sum()])


#循环训练
def training_loop(n_epochs,learning_rate,params,t_u,t_c):
    for epoch in range(1,n_epochs + 1):
        #参数
        w,b=params

        t_p = model(t_u,w,b)
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_c, t_u, t_p, w, b)
        params = params - learning_rate * grad

        print('Epoch %d,Loss %f'%(epoch,float(loss)))
    return params
training_loop(n_epochs=5000,learning_rate=1e-2,params=torch.tensor([1.0,0.0]),t_u=t_u * 0.1,t_c=t_c)
t_p = model(t_u*0.1,*params)
fig = plt.figure(dpi=600)
plt.xlabel("F")
plt.ylabel("C")
plt.plot(t_u.numpy(),t_p.detach().numpy())
plt.plot(t_u.numpy(),t_c.numpy(),'o')
plt.show()













