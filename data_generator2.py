import glob
import cv2
import numpy as np
import os
from torch.utils.data import Dataset

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

def gen_patches(file_name):
    # read image
    img = cv2.imread(file_name, 0)  # gray scale
    # img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)  # 缩小时，为了避免出现波纹现象，推荐采用INTER_AREA 区域插值方法。
    patches = []
    patches.append(img)
    return patches


def datagenerator(data_dir, verbose=False):
    file_list = glob.glob(data_dir + '/*.jpg')  # get name list of all .png files
    # print(file_list)
    file_list.sort(key=lambda x: int(x.replace(data_dir+'\\', "").split('.')[0]))
    # file_list = os.listdir(data_dir)
    # file_list.sort(key=lambda x:int(x.split('.')[0]))#按照数字进行排序后按顺序读取文件夹下的图片
    # for filename in os.listdir(data_dir):
    #     print(filename)
    # print(file_list)
    # initrialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        # print(file_list[i])
        patch = gen_patches(file_list[i])
        # print(patch)
        data.append(patch)
        if verbose:
            print(str(i + 1) + '/' + str(len(file_list)) + ' is done ^_^')
    # print(type(data))
    data = np.array(data, dtype='uint8')
    # print(type(data))
    print(data.shape)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3], 1))
    # print(data.shape)
    # discard_n = len(data) - len(data) // batch_size * batch_size;
    # data = np.delete(data, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')
    print(data.shape)
    print(type(data))
    return data


if __name__ == '__main__':
    data = datagenerator(data_dir=r'D:\CNN_Denoised\data\original')
