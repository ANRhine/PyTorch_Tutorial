#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      PyEg01_Warm-up:numpy.py
Author:         Ruonan Yu
Date:           18-1-31 
-------------------------------------
"""
import numpy as np

N = 64
D_in = 1000
D_out = 10
H = 100
LR = 1e-6
EPOCH = 500

# create random input and output
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

for epoch in range(EPOCH):
    # Forward pass:compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # compute and print loss
    loss = np.square(y_pred - y).sum()
    print('Epoch:', epoch,
          '| Loss:',loss)

    # Backpropagation to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # update weights
    w1 -= LR * grad_w1
    w2 -= LR * grad_w2
