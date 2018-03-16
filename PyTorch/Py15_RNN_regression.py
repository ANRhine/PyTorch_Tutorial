#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      Py15_RNN_regression.py
Author:         Ruonan Yu
Date:           18-3-16 
-------------------------------------
"""
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

TIME_STEP = 10  # rnn time step
INPUT_SIZE = 1  # rnn input size
LR = 0.02


steps = np.linspace(0, np.pi * 2, dtype=np.float32)
x_np = np.sin(steps)  # float32 for converting torch FloatTensor
y_np = np.cos(steps)
plt.plot(steps, y_np, 'r-', label='target(cos)')
plt.plot(steps, x_np, 'b-', label='input(sin)')
plt.legend(loc='best')
plt.show()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,  # 1
            hidden_size=32,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True  # input & output will has batch size as is dimension. e.g.(batch,time_step,input_size)
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        '''
        x (batch,time_step,input_size)
        h_state (n_layer,batch,hidden_size)
        r_out (batch,time_step,hidden_size)
        '''
        r_out, h_state = self.rnn(x, h_state)

        outs = []  # save all prediction
        for time_step in range(r_out.size(1)):  # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state

    # def forward(self,x,h_state):
    #     r_out,h_state=self.rnn(x,h_state)
    #     r_out_reshaped=r_out.view(-1,32)
    #     outs=self.linear_layer(r_out_reshaped)
    #     outs=outs.view(-1,TIME_STEP,INPUT_SIZE)


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss()

h_state = None  # 要使用初始化hidden state, 可以设成None

plt.figure(1, figsize=(12, 5))
plt.ion()  # continuously plot

for step in range(60):
    start, end = step * np.pi, (step + 1) * np.pi  # time range
    # use sin predicts cos
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)  # float32 for converting torch FloatTensor
    y_np = np.cos(steps)

    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))  # shape(batch,time_steo,input_size)
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))

    prediction, h_state = rnn(x, h_state)  # rnn对于每个step的prediction,还有最后一个step的h_state
    h_state = Variable(h_state.data)  # This is important!!要把h_state重新包装一下才能放入下一个iteration,不然会报错

    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # plotting
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.05)

plt.ioff()
plt.show()
