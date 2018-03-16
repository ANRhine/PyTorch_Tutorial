#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      plt001.py
Author:         Ruonan Yu
Date:           18-1-26 
-------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(-1,1,50)
#y=2*x+1
y=x**2
plt.figure()
plt.plot(x,y)
plt.show()