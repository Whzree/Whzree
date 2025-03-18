#!/usr/bin/env python
# -*- coding:UTF-8 -*-
'''
@Project:pythonProject1
@File:tensor1.py
@IDE:pythonProject1
@Author:whz
@Date:2025/3/15 10:07

'''
import numpy as np
import torch
import imageio.v2 as imageio
import os
import csv

L = [[4,1],[3,2],[5,6]]
points = torch.tensor(L)

img_arr = imageio.imread('img/png/2.jpg')
# 768 * 1024 像素点(宽度:768,高度:1024,3通道红绿蓝)
#print(img_arr.shape)
#(768, 1024, 3)
img = torch.from_numpy(img_arr)
out = img.permute(2,0,1)
#print(out.shape)
#torch.Size([3, 768, 1024])

#创建一个多图像的张量
batch_size = 3
batch = torch.zeros(batch_size,3,256,256,dtype = torch.uint8)
data_dir = './img/image-cats'
filenames = [name for name in os.listdir(data_dir)
             if os.path.splitext(name)[-1] == '.png']
for i,filename in enumerate(filenames):
    img_arr = imageio.imread(os.path.join(data_dir,filename))
    img_t = torch.from_numpy(img_arr)
    img_t = img_t.permute(2,0,1)
    img_t = img_t[:3]
    batch[i] = img_t
#转换张量数据类型
batch = batch.float()
#uint8 最大为255最小为0,最大最小归一化为[0,1]
batch /= 255.0
# z-score归一化
#N,C,H,W
#获取 批次数据 batch 的通道数
n_channels = batch.shape[1]
#遍历每个通道
for c in range(n_channels):
    #选择所有样本:,的第c个通道,形状为N,H,W
    mean = torch.mean(batch[:,c])
    std = torch.std(batch[:,c])
    batch[:,c] = (batch[:,c] - mean) / std

wine_path = "./img/data/winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path,dtype=np.float32,delimiter=";",skiprows=1)
#文件路径
#指定数据类型为32位浮点数
#列分隔符为冒号(:)
#跳过文件的第一行(通常是标题行)
# print(wineq_numpy)
col_list = next(csv.reader(open(wine_path),delimiter=";"))
#调用迭代器的 next() 方法，读取 CSV 文件的第一行（即列名）。
#print(col_list)
wineq = torch.from_numpy(wineq_numpy)
#print(wineq.shape,wineq.dtype)

data = wineq[:,:-1]
target = wineq[:,-1]
#将分数视为整数向量
target = wineq[:,-1].long()
#独热编码
target_onehot = torch.zeros(target.shape[0],10)
target_onehot.scatter_(1,target.unsqueeze(1),1.0)

#dim = 1：沿着第1个维度（列方向）填充。

#index = target.unsqueeze(1)：

#target.unsqueeze(1)
#将target的形状从(N, )转换为(N, 1)，例如[2, 5, 9] → [[2], [5], [9]]。
#这些值表示每个样本的标签在列方向上的索引位置。
#1.0：将指定索引位置的值设为 1.0。

#z-score 归一化
data_mean = torch.mean(data,dim=0)
#按行方向求均值,也就是结果是每一列的均值.
data_var = torch.var(data,dim=0)
# 求均值和标准差

data_normalized = (data - data_mean) / torch.sqrt(data_var)

#寻找阈值
bad_indexes = target <= 3
#print(bad_indexes.shape,bad_indexes.dtype,bad_indexes.sum())
bad_data = data[bad_indexes]
bad_data = data[target <= 3]
mid_data = data[(target > 3) & (target < 7)]
good_data = data[target >= 7]

bad_mean = torch.mean(bad_data,dim=0)
mid_mean = torch.mean(mid_data,dim=0)
good_mean = torch.mean(good_data,dim=0)

total_sulfur_threshold = 141.83
total_sulfur_data = data[:,6]
#第七列,也就是二氧化硫的含量
#按照二氧化硫中间品质的平均值作为阈值
predicted_indexes = torch.lt(total_sulfur_data,total_sulfur_threshold)
#真正好酒的索引
actual_indexes = target > 5
n_matches = torch.sum(actual_indexes & predicted_indexes).item()
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indexes).item()
#print(n_matches,n_predicted,n_actual)
#print(n_matches/n_predicted,n_matches/n_actual)


#处理时间序列变为张量
bikes_numpy = np.loadtxt(
    './img/data/hour-fixed.csv',
    dtype=np.float32,
    delimiter=",",
    skiprows = 1,
    converters={1:lambda x:float(x[8:10])}
    )
bikes = torch.from_numpy(bikes_numpy)
daily_bikes = bikes.view(-1,24,bikes.shape[1])
daily_bikes = daily_bikes.transpose(1,2)

#view 将原始张量重塑为 按天分组的三维张量
#-1:自动计算该维度的大小(总样本数需要被 24 整除)
# 24将每天的数据划分为24小时。
# [1]:保留原始特征数,即 特征数17
#准备训练
first_day = bikes[:24].long()
weather_onehot = torch.zeros(first_day.shape[0],4)
#print(first_day[:,9])
#根据对应的级别将1映射到矩阵中。
weather_onehot.scatter_(
    dim=1,
    index = first_day[:,9].unsqueeze(1).long() -1,
    value = 1.0
)

#print(weather_onehot)
torch.cat((bikes[:24],weather_onehot),1)[:1]











