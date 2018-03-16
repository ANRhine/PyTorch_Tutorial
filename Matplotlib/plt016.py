#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      plt016.py
Author:         Ruonan Yu
Date:           18-1-27 
-------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np

x=np.arange(0,10,0.1)
y1=0.05*x**2
y2=-1*y1

fig=plt.figure()
fig,ax1=plt.subplots()
ax2=ax1.twinx()

ax1.plot(x,y1,'g')
ax1.set_xlabel('X')
ax1.set_ylabel('Y1',color='g')

ax2.plot(x,y2,'b--')
ax2.set_ylabel('Y2',color='b')

plt.show()