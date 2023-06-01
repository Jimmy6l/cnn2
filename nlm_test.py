#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
# filePath = r'F:\主题一：遥感图像场景分类\val\val\水田\paddy-field_00076.jpg'
# img = cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
filePath = r'D:\CV\论文\1-硕士毕业论文\图片\第五章\5.3模拟斑点噪声图像\0.4.png'
img = cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
dst = cv2.fastNlMeansDenoising(img, None, 25,  3, 21)

# plt.subplot(121), plt.imshow(img)
# plt.subplot(122), plt.imshow(dst)
# plt.show()
cv2.imshow("image", dst) # 显示图片，后面会讲解
cv2.waitKey(0) #等待按键
cv2.imwrite('nlm0.4_new.png', dst)