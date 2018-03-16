#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      plt013.py
Author:         Ruonan Yu
Date:           18-1-26 
-------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np

plt.figure(1,figsize=(8,5))
plt.subplot(2,1,1)
plt.plot([0,1],[0,1])
plt.subplot(2,3,4)
plt.plot([0,1],[0,1])
plt.subplot(2,3,5)
plt.plot([0,1],[0,1])
plt.subplot(2,3,6)
plt.plot([0,1],[0,1])

plt.show()