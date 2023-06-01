import numpy as np
import os
import matplotlib.pyplot as plt
from keras_preprocessing import image
import cv2
import re

def resize_images(src_path, sizeX, sizeY):#对图像进行resize
    images = []
    imagesresize = []
    width, height = sizeX, sizeY
    for filename in os.listdir(src_path):
        img = cv2.imread(src_path + filename)
        if img is not None:
            images.append(img)
    cnt = 1
    for img in images:
        im2 = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

        ext = ".png"
        imagesresize.append(im2)
        cnt += 1
    print(len(imagesresize))
    return imagesresize

x = resize_images('J:\\DATA\\TEST\\SET1\\noise\\', 256,475) #载入数据
image_data = []

for file in x:#读取文件夹下的图片并保存到列表中
    gray_img = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
    img_array = image.img_to_array(gray_img)
    image_data.append(img_array)

img_data = np.array(image_data, dtype='float32') / 255.0
# print("111")
# print(img_data.shape)
# noisy_set = gaussian_noise(img_data,0,25)
noisy_set=img_data

'''
for i in range(noisy_set.shape[0]):
    cv2.imwrite("./noise/" + str(i) + ".jpg", noisy_set[i] * 255.)
'''
# print(noisy_set.shape)

from keras.layers import *
from keras.layers import Lambda
from keras.models import Model
from keras import backend as K


def DnCNN(depth, filters=64, image_channels=1, use_bnorm=True):
    layer_count = 0
    inpt = Input(shape=(None, None, image_channels), name='input' + str(layer_count))
    # 1st layer, Conv+relu
    layer_count += 1
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
               name='conv' + str(layer_count))(inpt)
    layer_count += 1
    x = Activation('relu', name='relu' + str(layer_count))(x)
    # depth-2 layers, Conv+BN+relu
    for i in range(depth - 2):
        layer_count += 1
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
                   use_bias=False, name='conv' + str(layer_count))(x)
        if use_bnorm:
            layer_count += 1
            # x = BatchNormalization(axis=3, momentum=0.1,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
        x = BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='bn' + str(layer_count))(x)
        layer_count += 1
        x = Activation('relu', name='relu' + str(layer_count))(x)
        # last layer, Conv
    layer_count += 1
    x = Conv2D(filters=image_channels, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal',
               padding='same', use_bias=False, name='conv' + str(layer_count))(x)
    layer_count += 1
    x = Subtract(name='subtract' + str(layer_count))([inpt, x])  # input - noise
    model = Model(inputs=inpt, outputs=x)

    return model

if __name__ == '__main__':
    bmodel =  DnCNN(depth=17,filters=64,image_channels=1,use_bnorm=True)
    #bmodel.load_weights('./save_weights/get_cnn_9conv25.h5')
    #bmodel.load_weights('./save_weights/get_cnn_architecture25.h5')
    bmodel.load_weights('./save_weights/202111_Laplacian.h5')

    pic = bmodel.predict(noisy_set)
    print("------------")
    print(noisy_set.shape)

    print(pic.shape)
    print(img_data[1].shape, pic[1].shape)

    # from skimage.measure import compare_psnr
    # psnrs = []
    # psnr1 = []
    # print(img_data[1].shape,pic[1].shape)
    #
    # for x in range(len(noisy_set)):
    #     '''
    #     psnr_x = skimage.metrics.peak_signal_noise_ratio(img_data[x] * 255., pic[x] * 255.,data_range = 255)
    #     psnr_y = skimage.metrics.peak_signal_noise_ratio(img_data[x] * 255., noisy_set[x] * 255.,data_range = 255)
    #     '''
    #     psnr_x = compare_psnr(img_data[x] * 255., pic[x] * 255., data_range=255)
    #     psnr_y = compare_psnr(img_data[x] * 255., noisy_set[x] * 255., data_range=255)
    #     psnrs.append(psnr_x)
    #     psnr1.append(psnr_y)
    #
    # psnr_avg = np.mean(psnrs)
    # psnr_avg1 = np.mean(psnr1)
    #
    # for y in psnrs:
    #     print(y, end=' ')
    #
    # print(psnr_avg)
    #
    # for z in psnr1:
    #     print(z, end=' ')

    # print(psnr_avg1)
    plt.figure('xiaoguo')
    plt.subplot(1,3,1)
    plt.imshow((img_data[2] * 255.).reshape(256,256), cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow((noisy_set[2] * 255.).reshape(256,256), cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow((pic[2] * 255.).reshape(256,256), cmap='gray')
    # cv2.imwrite('2.png', (pic[1] * 255.).reshape(256,256))
    plt.show()