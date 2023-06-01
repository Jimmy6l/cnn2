#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# %斑点噪声模拟实现
import cv2
import numpy as np
import numpy.matlib
import math

# 添加斑点噪声步骤：
# 首先产生和该图像大小一样的实部和虚部都是均值为0方差为σ^2的复数矩阵;
# 然后对该复数矩阵使用 3 × 3 的滑动窗口进行均值滤波;
# 最后使用式(1)的噪声模型对生成的幻影图像进行加噪处理来模拟超声图像的斑点噪声，这里γ取值为0.5。
# 在污染图像时，本文设置了四种噪声级别σ={ 5，10，15，20}进行测试。
# 参考：基于贝叶斯非局部平均滤波的超声图像斑点噪声抑制算法

mean = 0;
var = 0.01;  # σ^2
# print(var ** 0.5)
p = cv2.imread('1_00000226.png');  # ,cv2.IMREAD_GRAYSCALE
print(type(p))
# p = np.array(p / 255, dtype=float)
p = np.array(p / 255, dtype=float)
# print(p.shape)
# p=double(p)/255;
# 标准差为方差σ^2的平方根
# noise = np.random.normal(mean, var ** 0.5, p.shape)
noise = np.random.normal(mean, var** 0.5 , p.shape)
# nn = 0.4*randn(size(p))+0; #添加均值为0，方差为0.2,0.4,0.8的乘性
# % imhist(nn)

# noise = cv2.blur(noise, (3, 3))
J = p + np.sqrt(p) * noise;
'''
将值限制在(-1/0,1)间，然后乘255恢复
'''
if J.min() < 0:
    low_clip = -1.
else:
    low_clip = 0.

J = np.clip(J, low_clip, 1.0)
# J = J * 255;
J = np.uint8(J * 255);
gray2 = cv2.cvtColor(J, cv2.COLOR_BGR2GRAY)
# figure;
cv2.imshow('out', J);
cv2.imshow('out2', gray2);
cv2.waitKey(0) & 0xff

cv2.destroyAllWindows()
