import argparse
from keras.layers import *
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Subtract
from keras.models import Model
from keras.optimizers import Adam
import data_generator1 as dg
import keras.backend as K
import numpy as np
import cv2



## Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--train_data', default='D:\\CNN_Denoised\\OBNLM\\2021test2', type=str, help='path of train data')
parser.add_argument('--noise_data', default='D:\\CNN_Denoised\\OBNLM\\2021test2laplacian', type=str, help='path of noise data')

parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=100, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=1, type=int, help='save model at every x epoches')
args = parser.parse_args()
# print('parser')
# print(parser)


# DnCNN模型架构搭建
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


def train_datagen(epoch_iter=2000, epoch_num=5, batch_size=2, data_dir=args.train_data, data_dir2=args.noise_data):
    while (True):
        n_count = 0
        if n_count == 0:
            # print(n_count)
            xs = dg.datagenerator(data_dir)  # 数据增强
            xs = xs.astype('float32') / 255.0
            labels=dg.datagenerator(data_dir2)
            labels=labels.astype('float32') / 255.0
            # imageNumber = 1537
            # image_path = 'D:\\CNN_Denoised\\data\\DATA\\original\\'
            # labels = []
            # for i in range(imageNumber):
            #     label = cv2.imread(image_path + str(i + 1) + '.png', 0)
            #     labels.append(label)
            # labels = np.array(labels)
            # labels = labels.astype('float32') / 255.0
            # labels = np.expand_dims(labels, 3)
            indices = list(range(xs.shape[
                                     0]))  # 转换成列表
            # print('indices')
            # print(indices)
            n_count = 1

        for _ in range(epoch_num):
            np.random.shuffle(indices)  # shuffle
            for i in range(0, len(indices), batch_size):
                batch_x = xs[indices[i:i + batch_size]]
                batch_y = labels[indices[i:i + batch_size]]
                # noise = np.random.normal(0, args.sigma / 255.0, batch_x.shape)  # noise
                # noise =  K.random_normal(ge_batch_y.shape, mean=0, stddev=args.sigma/255.0)
                # batch_y = batch_x + noise #添加噪声

                yield batch_y, batch_x


# define loss
def sum_squared_error(y_true, y_pred):
    # return K.mean(K.square(y_pred - y_true), axis=-1)
    # return K.sum(K.square(y_pred - y_true), axis=-1)/2
    return K.sum(K.square(y_pred - y_true)) / 2


if __name__ == '__main__':
    model = DnCNN(depth=17, filters=64, image_channels=1, use_bnorm=True)

    # compile the model
    # model.compile(optimizer=Adam(args.lr), loss='mean_squared_error')
    model.compile(optimizer=Adam(args.lr), loss=sum_squared_error)

    model.fit_generator(train_datagen(batch_size=args.batch_size),
                        steps_per_epoch=2000, epochs=args.epoch, verbose=1)
    model.summary()

    # model.save('./save_weights/lw_model1.h5')
    model.save_weights('./save_weights/202111_Laplacian.h5')
