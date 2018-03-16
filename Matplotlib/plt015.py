#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      plt015.py
Author:         Ruonan Yu
Date:           18-1-27 
-------------------------------------
"""

import matplotlib.pyplot as plt

fig=plt.figure()
x=[1,2,3,4,5,6,7]
y=[1,3,4,5,7,8,3]

left,bottom,width,height=0.1,0.1,0.8,0.8
ax1=fig.add_axes([left,bottom,width,height])
ax1.plot(x,y,c='r')
ax1.set_title('title')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

left,bottom,width,height=0.15,0.6,0.25,0.25
ax1=fig.add_axes([left,bottom,width,height])
ax1.plot(x,y,c='b')
ax1.set_title('title inside 1')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

plt.axes([0.6,0.15,0.25,0.25])
plt.plot(y[::-1],x,'g')
plt.xlabel('x')
plt.ylabel("y")
plt.title('title inside 2')


plt.show()