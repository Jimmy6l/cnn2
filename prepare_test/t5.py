#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import cv2
import PSNR

img = cv2.imread("3_1.png", cv2.IMREAD_GRAYSCALE)
img0 = cv2.imread("Final3_0.png", cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread("BM3D_step2_demo.jpg", cv2.IMREAD_GRAYSCALE)
psnr = PSNR.PSNR(img, img0)
print("The PSNR between the two img of the First way is %f" % psnr)
psnr = PSNR.PSNR(img, img1)
print("The PSNR between the two img of the Second way is %f" % psnr)
