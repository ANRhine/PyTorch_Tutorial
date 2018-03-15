#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      Py02_classify.py
Author:         Ruonan Yu
Date:           18-1-28
-------------------------------------
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# *****************建立数据集****************
n_data = torch.ones(100, 2)  # n_date (tensor) shape(100,2)
x0 = torch.normal(2 * n_data, 1)  # normal(means.std) class0 x data (tensor),shape=(100,2)
y0 = torch.zeros(100)  # class0 y data (tensor),shape=(100,1)
x1 = torch.normal(-2 * n_data, 1)  # class1 x data (tensor),shape=(100,2)
y1 = torch.ones(100)  # class1 y data (tensor),shape=(100,1)

# 注意x,y数据的的数据形式一定要像下面一样(torch.cat是在合并数据)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # cat(seq,dim=0)　shape(200,2) FloatTensor=32-bit floating
y = torch.cat((y0, y1), 0).type(torch.LongTensor)  # shape(200,2) LongTensor=64-bit integer

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)  # tensor->Variable,只有Variable才可以计算梯度


# *****************建立神经网络***************
class Net(torch.nn.Module):  # 继承torch的Module
    # 定义所有层的属性
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 继承__init__的功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

    #  搭建层与层之间的关系
    def forward(self, x):  # 这同时也是Module中forward功能
        # 正向传播输出值，神经网络分析出输出值
        x = F.relu(self.hidden(x))  # 激励函数(隐藏层的线性值)
        x = self.predict(x)  # 输出值，不用激励函数
        return x


# 定义net
net = Net(2, 10, 2)
print(net)

# 快速搭建法
# net=torch.nn.Sequential(
#     torch.nn.Linear(2,10),
#     torch.nn.ReLU(),
#     torch.nn.Linear(10,2),
# )
# print(net)

# 使用ion()命令开启交互模式
plt.ion()  # 画图
plt.show()

# *****************训练网络*****************
# optimizer是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # 传入net的所有参数，学习率
loss_func = torch.nn.CrossEntropyLoss()  # the target label is Not an one-hotted

for t in range(100):
    out = net(x)  # 喂点net训练数据x，输出预测值

    loss = loss_func(out, y)  # 计算两者的误差

    optimizer.zero_grad()  # 清空上一步的残余更新参数值
    loss.backward()  # 误差反向传播，计算参数更新值
    optimizer.step()  # 将参数更新值施加到net的parameters上

    # ******************可视化训练过程**************
    # 　每2次输出一次
    if t % 2 == 0:
        plt.cla()  # clear axis
        # 过了一道softmax的激励函数后的最大概率才是预测值
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy().squeeze()  # Variable->tensor->numpy array->scalar
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200
        plt.text(1.5, -4, r'$accuracy=%.2f$' % accuracy, fontdict={'size': 10, 'color': 'red'})
        plt.pause(0.1)

# 关闭交互模式，防止图像一闪而过
plt.ioff()
plt.show()
