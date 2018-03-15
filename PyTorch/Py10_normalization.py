#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      Py10_normalization.py
Author:         Ruonan Yu
Date:           18-1-30 
-------------------------------------
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.nn import init
import torch.utils.data as Data

# hyper parameters
N_SAMPLES = 2000
BATCH_SIZE = 64
EPOCH = 12
LR = 0.03
N_HIDDEN = 8
ACTIVATION = F.tanh
B_INIT = -0.2  # 模拟不好的参数初始化

# training data
x = np.linspace(-7, 10, N_SAMPLES)[:, np.newaxis]
noise = np.random.normal(0, 2, x.shape)
y = np.square(x) - 5 + noise

# test data
test_x = np.linspace(-7, 10, 200)[:, np.newaxis]
test_noise = np.random.normal(0, 2, test_x.shape)
test_y = np.square(test_x) - 5 + test_noise

train_x, train_y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
test_x = Variable(torch.from_numpy(test_x).float(), volatile=True)  # volatile=True 不进行梯度更新
test_y = Variable(torch.from_numpy(test_y).float(), volatile=True)

train_data = Data.TensorDataset(data_tensor=train_x, target_tensor=train_y)
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)


# #show data
# plt.scatter(train_x.numpy(),train_y.numpy(),c='brown',s=50,alpha=0.2,label='train')
# plt.legend(loc='upper left')
# plt.show()

# 搭建neural network
class Net(nn.Module):
    def __init__(self, batch_normalization=False):
        super(Net, self).__init__()
        self.do_bn = batch_normalization
        self.fcs = []  # 太多层了，用for loop建立
        self.bns = []
        self.bn_input = nn.BatchNorm1d(1, momentum=0.5)  # 给input的BN,momentum用来平滑化batch mean and stddev

        for i in range(N_HIDDEN):  # 建层
            input_size = 1 if i == 0 else 10
            fc = nn.Linear(input_size, 10)
            setattr(self, 'fc%i' % i, fc)  # 注意！pytorch一定要你将层信息变成class的属性
            self._set_init(fc)
            self.fcs.append(fc)
            if self.do_bn:
                bn = nn.BatchNorm1d(10, momentum=0.5)
                setattr(self, 'bn%i' % i, bn)
                self.bns.append(bn)

        self.predict = nn.Linear(10, 1)  # output layer
        self._set_init(self.predict)  # 参数初始化

    def _set_init(self, layer):
        init.normal(layer.weight, mean=0., std=.1)
        init.constant(layer.bias, B_INIT)

    def forward(self, x):
        pre_activation = [x]
        if self.do_bn: x = self.bn_input(x)  # 判断是否要加BN
        layer_input = [x]
        for i in range(N_HIDDEN):
            x = self.fcs[i](x)
            pre_activation.append(x)  # 为之后出图
            if self.do_bn: x = self.bns[i](x)  # 判断是否要加BN
            x = ACTIVATION(x)
            layer_input.append(x)  # 为之后出图
        out = self.predict(x)
        return out, layer_input, pre_activation  # 建立两个net,一个有BN,一个没有


nets = [Net(batch_normalization=False), Net(batch_normalization=True)]
# print(*nets)

opts = [torch.optim.Adam(net.parameters(), lr=LR) for net in nets]

loss_func = torch.nn.MSELoss()

f, axs = plt.subplots(4, N_HIDDEN + 1, figsize=(10, 5))
plt.ion()  # something about plotting
plt.show()


def plot_histogram(l_in, l_in_bn, pre_ac, pre_ac_bn):
    for i, (ax_pa, ax_pa_bn, ax, ax_bn) in enumerate(zip(axs[0, :], axs[1, :], axs[2, :], axs[3, :])):
        [a.clear() for a in [ax_pa, ax_pa_bn, ax, ax_bn]]
        if i == 0:
            p_range = (-7, 10);
            the_range = (-7, 10)
        else:
            p_range = (-4, 4);
            the_range = (-1, 1)
        ax_pa.set_title('L' + str(i))
        ax_pa.hist(pre_ac[i].data.numpy().ravel(), bins=10, range=p_range, color='#FF9359', alpha=0.5);
        ax_pa_bn.hist(pre_ac_bn[i].data.numpy().ravel(), bins=10, range=p_range, color='#74BCFF', alpha=0.5)
        ax.hist(l_in[i].data.numpy().ravel(), bins=10, range=the_range, color='#FF9359');
        ax_bn.hist(l_in_bn[i].data.numpy().ravel(), bins=10, range=the_range, color='#74BCFF')
        for a in [ax_pa, ax, ax_pa_bn, ax_bn]: a.set_yticks(());a.set_xticks(())
        ax_pa_bn.set_xticks(p_range);
        ax_bn.set_xticks(the_range)
        axs[0, 0].set_ylabel('PreAct');
        axs[1, 0].set_ylabel('BN PreAct');
        axs[2, 0].set_ylabel('Act');
        axs[3, 0].set_ylabel('BN Act')
    plt.pause(0.01)


# training
losses = [[], []]  # 每个网络一个list来记录误差
for epoch in range(EPOCH):
    print('Epoch: ', epoch)
    layer_inputs, pre_acts = [], []
    for net, l in zip(nets, losses):
        net.eval()  # 一定要把Net的设置成eval模式，eval的BN参数会被固定
        pred, layer_input, pre_act = net(test_x)
        l.append(loss_func(pred, test_y).data[0])
        layer_inputs.append(layer_input)
        pre_acts.append(pre_act)
        net.train()  # 收集好信息后将net设置成train模式,继续训练
    plot_histogram(*layer_inputs, *pre_acts)  # plot histogram

    for step, (b_x, b_y) in enumerate(train_loader):
        b_x, b_y = Variable(b_x), Variable(b_y)
        for net, opt in zip(nets, opts):  # train for each network
            pred, _, _ = net(b_x)
            loss = loss_func(pred, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()  # it will also learns the parameters in Batch Normalization

plt.ioff()

# plot training loss
plt.figure(2)
plt.plot(losses[0], c='#FF9359', lw=3, label='Original')
plt.plot(losses[1], c='#74BCFF', lw=3, label='Batch Normalization')
plt.xlabel('step');
plt.ylabel('test loss');
plt.ylim((0, 2000));
plt.legend(loc='best')

# evaluation
# set net to eval mode to freeze the parameters in batch normalization layers
[net.eval() for net in nets]  # set eval mode to fix moving_mean and moving_var
preds = [net(test_x)[0] for net in nets]
plt.figure(3)
plt.plot(test_x.data.numpy(), preds[0].data.numpy(), c='#FF9359', lw=4, label='Original')
plt.plot(test_x.data.numpy(), preds[1].data.numpy(), c='#74BCFF', lw=4, label='Batch Normalization')
plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='r', s=50, alpha=0.2, label='train')
plt.legend(loc='best')
plt.show()
