import os
import cv2
import sys
import numpy as np

path = r"D:\CNN_Denoised\data\noise2clean\trainB\\"
newpath = r"D:\CNN_Denoised\data\noise2clean\t1\\"
print(path)

for filename in os.listdir(path):
    if os.path.splitext(filename)[1] == '.png':
        # print(filename)
        img = cv2.imread(path + filename)
        print(filename.replace(".png", ".jpg"))
        newfilename = filename.replace(".png", ".jpg")
        # cv2.imshow("Image",img)
        # cv2.waitKey(0)
        cv2.imwrite(newpath + newfilename, img)