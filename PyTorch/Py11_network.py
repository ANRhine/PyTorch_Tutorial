#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      Py11_network.py
Author:         Ruonan Yu
Date:           18-1-31 
-------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # (1,20,20)->(6,20,20)->(6,10,10)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # (6,10,10)->(16,10,10)->(16,5,5)
        x = x.view(-1, self.num_flat_features(x))  # (16,5,5)->(16,25)
        # x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))  # (16,25)->120
        x = F.relu(self.fc2(x))  # 120->84
        x = self.fc3(x)  # 84->10

        def num_flat_features(self, x):
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features


net1 = Net1()
print(net1)


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Sequential(  # (1,20,20)
            nn.Conv2d(1, 6, 5, 1, 2),  # ->(6,20,20)
            nn.ReLU(),
            nn.MaxPool2d(2)  # ->(6,10,10)
        )
        self.conv2 = nn.Sequential(  # (6,10,10)
            nn.Conv2d(6, 16, 5, 1, 2),  # ->(16,10,10)
            nn.ReLU(),
            nn.MaxPool2d(2)  # ->(16,5,5)
        )
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # (16,25)->120
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forword(self, x):
        x = self.conv1(x)  # (6,10,10)
        x = self.conv2(x)  # (16,5,5)
        # the size -1 is inferred from other dimensions
        x = x.view(x.size(0), -1)  # (16,5,5)->(16,25)
        x = F.relu(self.fc1(x))  # (16,25)->120
        x = F.relu(self.fc2(x) ) # 120->84
        x = self.fc3(x)  # 84->10
        return x


net2 = Net2()
# print(net2)

params = list(net2.parameters())
# print(len(params))
# print(params[0].size())  # conv1's .weight

input=Variable(torch.randn(1,1,32,32))
out=net2.forward(input)
print(out)

net2.zero_grad()
out.backward(torch.randn(1,10))
