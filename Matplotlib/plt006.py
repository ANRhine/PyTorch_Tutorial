#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      plt006.py
Author:         Ruonan Yu
Date:           18-1-26 
-------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-2,2,50)
y=x*2+1

plt.figure(1,figsize=(8,5))
plt.plot(x,y)

ax=plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))

x0=1
y0=2*x0+1
#散点图
plt.scatter(x0,y0,s=50,color='red')
plt.plot([x0,x0],[0,y0],'k--',linewidth=2.5)

#method 1
plt.annotate(r'$2x+1=%s$'%y0,xy=(x0,y0),xycoords='data',
             xytext=(+30,-30),textcoords='offset points',
             fontsize=16,arrowprops=dict(arrowstyle='->',
                                         connectionstyle="arc3,rad=0.2"))

#method 2
plt.text(-2,3,r'$This\ is\ some\ text.\mu\ \sigma_{i}\ \alpha_{t}$',
         fontdict={'size':14,'color':'r'})

plt.show()