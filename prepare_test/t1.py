#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import cv2
img=cv2.imread("3.png",0)
print(img.shape)
img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)
print(img.shape)
cv2.imshow("img",img)
cv2.waitKey(0) #等待按键
cv2.imwrite("3_1.png", img)
cv2.destroyAllWindows()
# img = cv2.imread("Final3_0.png", cv2.IMREAD_GRAYSCALE)  # 读入图像，cv2.IMREAD_GRAYSCALE:以灰度模式读入图像
# img = cv2.resize(img, (160, 512), interpolation=cv2.INTER_AREA)
# cv2.imwrite("original3_0.png", img)
