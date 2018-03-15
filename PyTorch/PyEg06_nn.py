#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      PyEg06_nn.py
Author:         Ruonan Yu
Date:           18-2-1 
-------------------------------------
"""
import torch
from torch.autograd import Variable

N = 64
D_in = 1000
D_out = 10
H = 100
LR = 1e-6
EPOCH = 500

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

loss_fn = torch.nn.MSELoss(size_average=False)

for epoch in range(EPOCH):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print('Epoch:', epoch,
          '| Loss:', loss.data[0])
    model.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.data -= LR * param.grad.data
