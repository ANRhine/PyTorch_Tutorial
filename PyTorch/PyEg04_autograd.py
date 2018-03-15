#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      PyEg04_autograd.py
Author:         Ruonan Yu
Date:           18-2-1 
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


class MyReLU(torch.autograd.Function):
    def forward(self, input):
        self.save_for_backward(input)
        return input.clamp(min=0)

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

for epoch in range(EPOCH):
    relu = MyReLU()

    y_pred = relu(x.mm(w1)).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    print('Epoch:', epoch,
          '| Loss:', loss.data[0])
    loss.backward()

    w1.data -= LR * w1.grad.data
    w2.data -= LR * w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()
