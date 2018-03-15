#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      PyEg03_Variables&autograd.py
Author:         Ruonan Yu
Date:           18-1-31 
-------------------------------------
"""
import torch
from torch.autograd import Variable

dtype = torch.cuda.FloatTensor

N = 64
D_in = 1000
D_out = 10
H = 100
LR = 1e-6
EPOCH = 500

x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=True)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=True)

w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

for epoch in range(EPOCH):
    pred_y = x.mm(w1).clamp(min=0).mm(w2)
    loss = (pred_y - y).pow(2).sum()
    print('Epoch:', epoch,
          '| Loss:', loss)

    loss.backward()
    w1.data -= LR * w1.grad.data
    w2.data -= LR * w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()
