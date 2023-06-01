#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import cv2
img=cv2.imread("BM3D_step2_demo.jpg",cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (467, 467), interpolation=cv2.INTER_CUBIC) #放大图像，通常使用INTER_CUBIC(速度较慢，但效果最好)
img = cv2.resize(img, (256, 467), interpolation=cv2.INTER_AREA) #缩小时，为了避免出现波纹现象，推荐采用INTER_AREA 区域插值方法。
cv2.imshow("img",img)
cv2.imwrite("res1_1.png", img)
cv2.waitKey(0) #等待按键
cv2.destroyAllWindows()