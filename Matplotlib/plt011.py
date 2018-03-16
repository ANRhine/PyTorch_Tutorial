#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
-------------------------------------
File name:      plt011.py
Author:         Ruonan Yu
Date:           18-1-26 
-------------------------------------
"""
import matplotlib.pyplot as plt
import numpy as np

#image data
a=np.array([0.313660827978, 0.365348418405, 0.423733120134,
              0.365348418405, 0.439599930621, 0.525083754405,
              0.423733120134, 0.525083754405, 0.651536351379]).reshape(3,3)

#https://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html
plt.imshow(a,interpolation='none',cmap='bone',origin='upper')
plt.colorbar(shrink=0.9)

plt.xticks(())
plt.yticks(())
plt.show()