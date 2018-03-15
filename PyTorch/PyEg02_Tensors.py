#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      PyEg02_Tensors.py
Author:         Ruonan Yu
Date:           18-1-31 
-------------------------------------
"""
import torch

dtype = torch.cuda.FloatTensor

N = 64
D_in = 1000
D_out = 10
H = 100
LR = 1e-6
EPOCH = 500

x = torch.randn(N, D_in).type(dtype)
y = torch.randn(N, D_out).type(dtype)

w1 = torch.randn(D_in, H).type(dtype)
w2 = torch.randn(H, D_out).type(dtype)

for epoch in range(EPOCH):
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    loss = (y_pred - y).pow(2).sum()
    print('Epoch:', epoch,
          '| Loss:', loss)

    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    w1 -= LR * grad_w1
    w2 -= LR * grad_w2
