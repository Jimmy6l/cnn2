#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# 图像差分-图像相减
import cv2
import numpy as np

img1 = cv2.imread(r'D:\Code_Data\0Proposed_DnCNN\data\test\resize\10.png')  # 噪声
img2 = cv2.imread(r'D:\Code_Data\0Proposed_DnCNN\results\resize\10_dscnn.png')  # 干净
img3 = cv2.imread(r'C:\Users\Jimmy\Desktop\2022testimg\denoise\my\10.png')

# img1 = img1.astype('float32')
img1 = np.array(img1, dtype=np.float32)
img2 = np.array(img2, dtype=np.float32)
# img2 = img2.astype('float32')
sub1 = np.subtract(img1, img2)  # !!!x1 和 x2 的差异，逐元素。
sub = cv2.convertScaleAbs(img1 - img2)
# sub1 = cv2.convertScaleAbs(img1 - sub)  # !!!
# sub = cv2.convertScaleAbs(sub)
img1 = cv2.convertScaleAbs(img1)
img2 = cv2.convertScaleAbs(img2)
add = cv2.convertScaleAbs(img2 + sub1)
# 方法二
err = cv2.absdiff(img1, img2)  # 差值的绝对值
err2 = cv2.absdiff(img1, img3)
cv2.imwrite("err1.png", err)
cv2.imwrite("err2.png", err2)
# err1 = cv2.absdiff(img1, err)  # 差值的绝对值
# err2 = img1 - err
print(err)
cv2.imshow('img', img1)
cv2.imshow('img_d', img2)
cv2.imshow('err', err)
cv2.imshow('err2', err2)
# cv2.imshow('err1', err1)
# cv2.imshow('sub1', sub1)
# cv2.imshow('err2', err2)
cv2.imshow('sub', sub)
# cv2.imshow('add', add)
# cv2.imwrite("grayimages/err1.png", err1)
# cv2.imwrite("grayimages/sub1.png", sub)

cv2.waitKey(0)
cv2.destroyAllWindows()
