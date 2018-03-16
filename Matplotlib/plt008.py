#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      plt008.py
Author:         Ruonan Yu
Date:           18-1-26 
-------------------------------------
"""
import matplotlib.pyplot as plt
import numpy as np

n=1024
X=np.random.normal(0,1,n)#随机生成均值为０，方差为１的数据ｘ
Y=np.random.normal(0,1,n)
color=np.arctan2(Y,X)#for color value

plt.scatter(X,Y,s=50,c=color,alpha=0.5)
#plt.scatter(np.arange(5),np.arange(5))

plt.xlim((-2,2))
plt.ylim((-2,2))
plt.xticks(())
plt.yticks(())


plt.show()