#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      PyEg08_Custom_nn_Modules.py
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
LR = 1e-4
EPOCH = 500

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

model = TwoLayerNet(D_in, H, D_out)

optimizer = torch.optim.SGD(model.parameters(), lr=LR)
loss_func = torch.nn.MSELoss(size_average=False)

for epoch in range(EPOCH):
    y_pred = model(x)
    loss = loss_func(y_pred, y)
    print('Epoch:', epoch,
          '| Loss:', loss.data[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
