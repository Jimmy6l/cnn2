import os
import cv2
import sys
import numpy as np

# path = r"D:\CNN_Denoised\data\noise2clean\trainB\\"
# newpath = r"D:\CNN_Denoised\data\noise2clean\t1\\"
# print(path)
path = r"E:\train\\"
newpath = r"E:\超声图像数据集\Ultrasound Nerve Segmentation\test_data\\"

for filename in os.listdir(path):
    if os.path.splitext(filename)[1] == '.tif':
        # print(filename)
        img = cv2.imread(path + filename)
        print(filename.replace(".tif", ".jpg"))
        newfilename = filename.replace(".tif", ".jpg")
        # cv2.imshow("Image",img)
        # cv2.waitKey(0)
        # cv2.imwrite(newpath + newfilename, img)
        cv2.imwrite(path + newfilename, img)