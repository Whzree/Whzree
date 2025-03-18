#!/usr/bin/env python
# -*- coding:UTF-8 -*-
'''
@Project:pythonProject1
@File:Char2.py
@IDE:pythonProject1
@Author:whz
@Date:2025/3/12 14:12

'''
from torchvision import transforms
from PIL import Image
#print(dir(models))
import torchvision.models as models
from torchvision.models.resnet import ResNet101_Weights
import torch

# 使用 weights 参数
resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
#print(resnet)
preprocess = transforms.Compose([
    #将图片大小转换为256 × 256 个像素
    transforms.Resize(256),
    #围绕中心将图像裁剪为 224 × 224 个像素
    transforms.CenterCrop(224),
    #将其转化为一个张量
    transforms.ToTensor(),
    #将其RGB分量（红色，绿色和蓝色）进行归一化处理，使其具有定义的均值和标准差。
    transforms.Normalize(
        mean = [0.485,0.456,0.406],
        std = [0.229,0.224,0.225]
    )
])
#张量是一种 PyTorch 多维数组，在本例中，是一个包含颜色，高度，宽度，的三维数组。
img = Image.open("img/png/2.jpg")
#img.show()
#图片进行重塑，裁剪
img_t = preprocess(img)
#归一化
batch_t = torch.unsqueeze(img_t,0)







#eval
resnet.eval()
out = resnet(batch_t)
#print(out)
with open('img/data/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]
_,index = torch.max(out,1)
percentage = torch.nn.functional.softmax(out,dim=1)[0]*100
print(labels[index[0]],percentage[index[0]].item())


