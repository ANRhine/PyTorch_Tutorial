#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      plt007.py
Author:         Ruonan Yu
Date:           18-1-26
-------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(-3,3,50)
y=0.1*x

plt.figure(1,figsize=(8,5))
plt.plot(x,y,lw=10)
plt.ylim((-2,2))
ax=plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))


for label in ax.get_xticklabels()+ax.get_yticklabels():
    label.set_fontsize(12)
    label.set_bbox(dict(facecolor='white',edgecolor='none',alpha=0.7))

plt.show()