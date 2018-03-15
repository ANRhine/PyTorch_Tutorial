#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      Py01_regression.py
Author:         Ruonan Yu
Date:           18-1-27 
-------------------------------------
"""

import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F  # 激活函数在此

# *****************建立数据集****************
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor),shape (100,1)
y = x ** 2 + 0.2 * torch.rand(x.size())  # noisy y data (tensor),shape=(100,1)

# 用Variable来修饰这些数据tensor
x, y = Variable(x), Variable(y)

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
net = Net(1, 10, 1)

# 快速搭建法
# net=torch.nn.Sequential(
#     torch.nn.Linear(1,10),
#     torch.nn.ReLU(),
#     torch.nn.Linear(10,1)
# )
# print(net)

# 使用ion()命令开启交互模式
plt.ion()
plt.show()

# *****************训练网络*****************
# optimizer是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  # 传入net的所有参数，学习率
loss_func = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式（均方误差）

for t in range(100):
    prediction = net(x)  # 喂点net训练数据x，输出预测值

    loss = loss_func(prediction, y)  # 计算两者的误差

    optimizer.zero_grad()  # 清空上一步的残余更新参数值
    loss.backward()  # 误差反向传播，计算参数更新值
    optimizer.step()  # 将参数更新值施加到net的parameters上

    # ******************可视化训练过程**************
    # 　每５次输出一次
    if t % 5 == 0:
        plt.cla()  # clear axis
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, r'$loss=%.4f$' % loss.data[0], fontdict={'size': 10, 'color': 'red'})
        plt.pause(0.1)

# 关闭交互模式，防止图像一闪而过
plt.ioff()
plt.show()
