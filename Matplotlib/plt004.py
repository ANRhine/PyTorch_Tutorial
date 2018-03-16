#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      plt004.py
Author:         Ruonan Yu
Date:           18-1-26 
-------------------------------------
"""
import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(-2,2,50)
y1=x*2+1
y2=x**2

plt.figure(3,figsize=(8,5))
plt.plot(x,y1)
plt.plot(x,y2,c='red',linestyle='--')

plt.xlim((-2,2))
plt.ylim((-1,3))
plt.xlabel('x')
plt.ylabel('y')

new_ticks=np.linspace(-2,2,5)
plt.xticks(new_ticks)
plt.yticks([-1,0,1,2,3],
           [r'$really\ bad$',r'$bad$',r'$normal$',r'$good$',r'$really\ good$'])

# gca=get current axis
ax=plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))

plt.show()
