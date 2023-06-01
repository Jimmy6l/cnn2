# -*- coding: utf-8 -*-

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26},
#    number={7},
#    pages={3142-3155},
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================

# no need to run this code separately


import glob

import cv2
import numpy as np
import torch
# from multiprocessing import Pool
from torch.utils.data import Dataset

patch_size, stride = 40, 10  # 补丁大小40   步长10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128  # 批量大小


# 加噪声类
class DenoisingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """

    def __init__(self, xs, noise, sigma):
        super(DenoisingDataset, self).__init__()
        self.xs = xs  # 清洁图像
        self.noise = noise
        self.sigma = sigma

    def __getitem__(self, index):
        batch_x = self.xs[index]
        batch_y = self.noise[index]
        # noise = torch.randn(batch_x.size()).mul_(self.sigma/255.0)
        # # #torch.randn：返回一个张量，包含了从区间[0,1)的均匀分布中抽取的一组随机数，形状由可变参数sizes 定义
        # batch_y = batch_x + noise #加噪声
        return batch_y, batch_x  # 返回批量batch_y, batch_x

    def __len__(self):
        return self.xs.size(0)


# 展示图片
def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def data_aug(img, mode=0):
    # data augmentation 数据增强
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)  # 翻转变换(flip): 沿着水平或者垂直方向翻转图像  #flipud(a) 上下翻转
    elif mode == 2:
        return np.rot90(img)  # 将矩阵A逆时针旋转90°以后返回
    elif mode == 3:
        return np.flipud(np.rot90(img))  # 先反转再旋转
    elif mode == 4:
        return np.rot90(img, k=2)  # 将矩阵逆时针旋转（90×k）°以后返回，k取负数时表示顺时针旋转，再翻转
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))  # 先旋转再翻转
    elif mode == 6:
        return np.rot90(img, k=3)  # 将矩阵逆时针旋转（90×k）°以后返回，k取负数时表示顺时针旋转
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))  # 先旋转再翻转


# 从一张图像中获取多尺度的补丁
def gen_patches(file_name):
    # get multiscale patches from a single image
    img = cv2.imread(file_name, 0)  # gray scale
    # patches = []
    # patches.append(img)
    # return patches
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h * s), int(w * s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled - patch_size + 1, stride):
            for j in range(0, w_scaled - patch_size + 1, stride):
                x = img_scaled[i:i + patch_size, j:j + patch_size]
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)
    return patches


# 从数据集中生成干净的补丁
def datagenerator(data_dir='data/Train400', verbose=False):
    # generate clean patches from a dataset
    print(data_dir)
    file_list = glob.glob(data_dir + '/*.png')  # get name list of all .png files
    file_list.sort(key=lambda x: int(x.replace(data_dir + '\\', "").split('.')[0]))
    # initrialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i])  # 调用自定义函数gen_patches
        # for patch in patches:
        data.append(patches)
        if verbose:
            print(str(i + 1) + '/' + str(len(file_list)) + ' is done ^_^')
    data = np.array(data, dtype='uint8')
    print(data.shape)
    # data=np.concatenate(data,axis=0) #ADD
    data = np.expand_dims(data, axis=3)  # np.expand_dims  扩展维度
    discard_n = len(data) - len(data) // batch_size * batch_size  # because of batch namalization
    # discard_n = len(data)-len(data)//batch_size*batch_size 这个就是判断要删除多少
    # 这个是因为batch_size的大小设置的是128,可能数据块长度不能恰好被整除，就删除多余的。
    data = np.delete(data, range(discard_n), axis=0)  # delete是可以删除数组的整行和整列的
    print('^_^-training data finished-^_^')
    return data


if __name__ == '__main__':
    data = datagenerator(data_dir=r'D:\CNN_Denoised\OBNLM\DATASET\noise')

#    print('Shape of result = ' + str(res.shape))
#    print('Saving data...')
#    if not os.path.exists(save_dir):
#            os.mkdir(save_dir)
#    np.save(save_dir+'clean_patches.npy', res)
#    print('Done.')
