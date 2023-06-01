import cv2
import os

def read_path(file_pathname):
    print(file_pathname)
    #遍历该目录下的所有图片文件
    for filename in os.listdir(file_pathname):
        print(filename)
        img = cv2.imread(file_pathname+'/'+filename)
        ####change to gray
      #（下面第一行是将RGB转成单通道灰度图，第二步是将单通道灰度图转成3通道灰度图）
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # image_np=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        #####save figure
        cv2.imwrite(file_pathname+"/"+'0_'+filename,img)

#注意*处如果包含家目录（home）不能写成~符号代替 
#必须要写成"/home"的格式，否则会报错说找不到对应的目录
#读取的目录
read_path("D:\BM3D_codes\BM3D-Denoise-master\grayimages")
#print(os.getcwd())
