#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      plt003.py
Author:         Ruonan Yu
Date:           18-1-26 
-------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(-1,1,50)
y1=x*2+1
y2=x**2

plt.figure(1,figsize=(8,5))
plt.plot(x,y1)
plt.plot(x,y2,c='red',linewidth=1.0,linestyle='--')

plt.xlim((-1,2))
plt.ylim((-2,3))
plt.xlabel('I am x')
plt.ylabel('I am y')

#换间隔
new_ticks=np.linspace(-1,2,5)
print(new_ticks)
plt.xticks(new_ticks)
plt.yticks([-2,-1.8,-1,1.22,3],
           [r'$really\ bad$',r'$bad\ \alpha$',r'$normal$',r'$good$',r'$really\ good$'])


plt.show()
