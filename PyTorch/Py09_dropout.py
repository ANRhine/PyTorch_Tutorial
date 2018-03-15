#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      Py09_dropout.py
Author:         Ruonan Yu
Date:           18-1-30 
-------------------------------------
"""

import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn

torch.manual_seed(1)

N_SAMPLES = 20
N_HIDDEN = 300
LR = 0.001

# fake data
# training data
x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
y = x + 0.3 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))
x, y = Variable(x), Variable(y)

# test data
test_x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
test_y = test_x + 0.3 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))
test_x, test_y = Variable(test_x, volatile=True), Variable(test_y, volatile=True)

# overfitting network
net_overfitting = nn.Sequential(
    nn.Linear(1, N_HIDDEN),
    nn.ReLU(),
    nn.Linear(N_HIDDEN, N_HIDDEN),
    nn.ReLU(),
    nn.Linear(N_HIDDEN, 1)
)

# dropout network
net_dropouted = nn.Sequential(
    nn.Linear(1, N_HIDDEN),
    nn.Dropout(0.5),  # drop 50% of neuron
    nn.ReLU(),
    nn.Linear(N_HIDDEN, N_HIDDEN),
    nn.Dropout(0.5),  # drop 50% of neuron
    nn.ReLU(),
    nn.Linear(N_HIDDEN, 1)
)

print(net_overfitting)
print(net_dropouted)

# training
optimizer_ofit = torch.optim.Adam(net_overfitting.parameters(), lr=LR)
optimizer_drop = torch.optim.Adam(net_dropouted.parameters(), lr=LR)
loss_func = nn.MSELoss()

plt.ion()

for t in range(500):
    pred_ofit = net_overfitting(x)
    pred_drop = net_dropouted(x)
    loss_ofit = loss_func(pred_ofit, y)
    loss_drop = loss_func(pred_drop, y)

    optimizer_ofit.zero_grad()
    optimizer_drop.zero_grad()
    loss_ofit.backward()
    loss_drop.backward()
    optimizer_ofit.step()
    optimizer_drop.step()

    if t % 10 == 0:  # 每10步画一次图
        # 将神经网络转换test形式，画好图之后改回训练形式
        net_overfitting.eval()
        net_dropouted.eval()  # 因为drop网络在train的时候和test的时候参数不一样

        plt.cla()
        test_pred_ofit = net_overfitting(test_x)
        test_pred_drop = net_dropouted(test_x)
        plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', alpha=0.5, label='train')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label='test')
        plt.plot(test_x.data.numpy(), test_pred_ofit.data.numpy(), 'r-', lw=3, label='overfitting')
        plt.plot(test_x.data.numpy(), test_pred_drop.data.numpy(), 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, r'$overfitting loss=%.4f$' % loss_func(test_pred_ofit, test_y).data[0],
                 fontdict={'size': 10, 'color': 'red'})
        plt.text(0, -1.5, r'$dropout loss=%.4f$' % loss_func(test_pred_drop, test_y).data[0],
                 fontdict={'size': 10, 'color': 'red'})
        plt.legend(loc='upper left')
        plt.ylim((-2.5, 2.5))
        plt.pause(0.1)

        # 将两个网络改回train形式
        net_overfitting.train()
        net_dropouted.train()

plt.ioff()
plt.show()
