#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      plt017.py
Author:         Ruonan Yu
Date:           18-1-27 
-------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

fig,ax=plt.subplots()

x=np.arange(0,4*np.pi,0.01)
line,=ax.plot(x,np.sin(x))

def animate(i):
    line.set_ydata(np.sin(x+i/10))
    return line,

def init():
    line.set_ydata(np.sin(x))
    return line,


ani=animation.FuncAnimation(fig=fig,func=animate,frames=100,init_func=init,interval=20,blit=True)

plt.show()