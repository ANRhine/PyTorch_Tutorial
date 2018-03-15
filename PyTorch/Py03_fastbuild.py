#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      Py03_fastbuild.py
Author:         Ruonan Yu
Date:           18-1-28 
-------------------------------------
"""
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable

n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), 0).type(torch.LongTensor)

x, y = Variable(x), Variable(y)


# ***************method 1************
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        # x = torch.nn.ReLU(self.hidden(x))
        x = self.predict(x)
        return x


net1 = Net(2, 10, 2)
print(net1)

# ****************method 2**************
net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2),
)
print(net2)

# plt.ion()
#
# optimizer = torch.optim.SGD(net1.parameters(), lr=0.02)
# loss_func = torch.nn.CrossEntropyLoss()
#
# for t in range(100):
#     out = net1(x)
#
#     loss = loss_func(out, y)
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if t % 2 == 0:
#         plt.cla()
#         prediction = torch.max(F.softmax(out), 1)[1]  # 把输出值转换成概率,用softmax
#         pred_y = prediction.data.numpy().squeeze()
#         target_y = y.data.numpy()
#         plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
#         accuracy = sum(pred_y == target_y) / 200
#         plt.text(1.5, -4, r'$accuracy=%.2f$' % accuracy, fontdict={'size': 10, 'color': 'red'})
#         plt.pause(0.1)
#
# plt.ioff()
# plt.show()
