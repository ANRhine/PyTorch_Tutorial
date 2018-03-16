#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      plt002.py
Author:         Ruonan Yu
Date:           18-1-26 
-------------------------------------
"""
import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(-1,1,50)
y1=x**2
y2=2*x+1

plt.figure()
plt.plot(x,y1)

plt.figure(3,figsize=(8,5))
plt.plot(x,y2)
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--')

plt.show()