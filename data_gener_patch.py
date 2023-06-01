import glob
import cv2
import numpy as np
# from multiprocessing import Pool
from torch.utils.data import Dataset
import torch

# patch_size, stride = 80, 20
# aug_times = 1
# scales = [1, 0.9, 0.8, 0.7]
# batch_size = 128
patch_size, stride = 80, 20
aug_times = 1
scales = [1]
batch_size = 8

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
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name):
    # get multiscale patches from a single image
    img = cv2.imread(file_name, 0)  # gray scale
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[i:i+patch_size, j:j+patch_size]
                for k in range(0, aug_times):
                    # x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    x_aug = data_aug(x, mode=0)
                    patches.append(x_aug)
    return patches


def datagenerator(data_dir='data/Train400', verbose=False):
    # generate clean patches from a dataset
    file_list = glob.glob(data_dir+'/*.jpg')  # get name list of all .png files
    # --
    file_list.sort(key=lambda x: int(x.replace(data_dir+'\\', "").split('.')[0]))
    # --
    # initrialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i])
        for patch in patches:
            data.append(patch)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list)) + ' is done ^_^')
    data = np.array(data, dtype='uint8')
    # --
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3], 1))
    # --
    # data = np.expand_dims(data, axis=3)
    # discard_n = len(data)-len(data)//batch_size*batch_size  # because of batch namalization
    # data = np.delete(data, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')
    return data


if __name__ == '__main__':

    data = datagenerator(data_dir='data/Train400')


#    print('Shape of result = ' + str(res.shape))
#    print('Saving data...')
#    if not os.path.exists(save_dir):
#            os.mkdir(save_dir)
#    np.save(save_dir+'clean_patches.npy', res)
#    print('Done.')