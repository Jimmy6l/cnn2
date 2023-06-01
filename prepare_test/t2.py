#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import pandas as pd

img1 = cv2.imread('3_0.png')
img2 = cv2.imread('First3_0.png')
img3 = cv2.imread('Final3_0.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
# ====使用numpy的数组矩阵合并concatenate======

# image = np.concatenate((gray1, gray2))
image = np.vstack((gray1, gray2,gray3))
# 纵向连接 image = np.vstack((gray1, gray2))
# 横向连接 image = np.concatenate([gray1, gray2], axis=1)
# image = np.array(df) # dataframe to ndarray

# =============
cv2.imwrite("contact3_1.png", image)
cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()