# 批量对文件夹的图片进行处理并保存

import cv2
import os


def read_path(file_pathname, new_pathname):
    # print(file_pathname)
    # 遍历该目录下的所有图片文件
    for filename in os.listdir(file_pathname):
        print(filename)
        img = cv2.imread(file_pathname + '/' + filename, cv2.IMREAD_GRAYSCALE)
        ####change to gray
        # （下面第一行是将RGB转成单通道灰度图，第二步是将单通道灰度图转成3通道灰度图）
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)   # 缩小时，为了避免出现波纹现象，推荐采用INTER_AREA 区域插值方法。
        # img = cv2.resize(img, (160, 475), interpolation=cv2.INTER_CUBIC)  # 放大图像，通常使用INTER_CUBIC(速度较慢，但效果最好)
        # img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # image_np=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        #####save figure
        cv2.imwrite(new_pathname + "/" + filename, img)


# 注意*处如果包含家目录（home）不能写成~符号代替
# 必须要写成"/home"的格式，否则会报错说找不到对应的目录
# 读取的目录
file_pathname = "E:\\train"
new_pathname = "E:\\train_resize"
read_path(file_pathname, new_pathname)
# print(os.getcwd())
