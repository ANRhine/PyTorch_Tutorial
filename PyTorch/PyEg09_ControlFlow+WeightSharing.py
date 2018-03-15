#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      PyEg09_ControlFlow+WeightSharing.py
Author:         Ruonan Yu
Date:           18-2-1 
-------------------------------------
"""

import random
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


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


model = DynamicNet(D_in, H, D_out)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
loss_func = torch.nn.MSELoss(size_average=False)
for epoch in range(EPOCH):
    y_pred = model(x)
    loss = loss_func(y_pred, y)
    print('Epoch:', epoch,
          '| Loss:', loss.data[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
