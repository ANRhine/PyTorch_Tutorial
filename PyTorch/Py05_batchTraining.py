#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      Py05_batchTraining.py
Author:         Ruonan Yu
Date:           18-1-29 
-------------------------------------
"""
import torch
import torch.utils.data as Data

BATCH_SIZE=5

x=torch.linspace(1,10,10)
y=torch.linspace(10,1,10)

#先转换成torch能识别的Dataset
torch_dataset=Data.TensorDataset(data_tensor=x,target_tensor=y)

#把dataset放入DataLoader
loader=Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2 #多线程来读数据
)

for epoch in range(5): #训练整套数据５次
    for step,(batch_x,batch_y) in enumerate(loader):#每一步loader释放一小批数据用来学习
        #training ...

        #打出来一些数据
        print('Epoch',epoch,'| Step',step,'| batch x',batch_x.numpy(),'| batch y',batch_y.numpy())