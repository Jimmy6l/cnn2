# -*- coding: utf-8 -*-
# PyTorch 0.4.1, https://pytorch.org/docs/stable/index.html
# =============================================================================
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================
# run this to train the model
# =============================================================================
# For batch normalization layer, momentum should be a value from [0.1, 1] rather than the default 0.1. 
# The Gaussian noise output helps to stablize the batch normalization, thus a large momentum (e.g., 0.95) is preferred.
# =============================================================================

import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import data_generator2 as dg
from data_generator2 import DenoisingDataset
# import data_generator1 as dg
# from data_generator1 import DenoisingDataset

# Params
# 创建一个解析器
# 使用 argparse 的第一步是创建一个 ArgumentParser 对象
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
# 给一个 ArgumentParser 添加程序参数信息是通过调用 add_argument() 方法完成的。
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
# 批量大小  整型   默认大小128
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
# 训练数据   字符串  默认 data/Train400  路径
parser.add_argument('--train_data', default='D:\\CNN_Denoised\\OBNLM\\DATASET\\clean', type=str,
                    help='path of train data')
parser.add_argument('--noise_data', default='D:\\CNN_Denoised\\OBNLM\\DATASET\\noise', type=str,
                    help='path of train data')
# 噪声水平  整型  默认25
parser.add_argument('--sigma', default=28, type=int, help='noise level')
# epoch 整型  默认180
parser.add_argument('--epoch', default=10, type=int, help='number of train epoches')
# 学习率  float 0.001  adam优化算法
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
# 解析参数
# ArgumentParser 通过 parse_args() 方法解析参数。
args = parser.parse_args()

batch_size = args.batch_size
cuda = torch.cuda.is_available()
n_epoch = args.epoch
sigma = args.sigma
# os.path.join()：  将多个路径组合后返回args.model = DNCNN  str(sigma)=25
# 组合之后的路径为models/DNCNN_sigma25
save_dir = os.path.join('models', args.model + '_' + 'sigma' + str(sigma))

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


# 定义构造函数
# 构建网络最开始写一个class，然后def _init_（输入的量），然后super（DnCNN，self）.__init__()这三句是程式化的存在，
# 初始化
class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3  # 卷积核的大小  3*3
        padding = 1  # padding表示的是在图片周围填充0的多少，padding=0表示不填充，padding=1四周都填充1维
        layers = []
        # 四个参数 输入的通道  输出的通道  卷积核大小  padding
        # 构建一个输入通道为channels，输出通道为64，卷积核大小为3*3,四周进行1个像素点的零填充的conv1层  #bias如果bias=True，添加偏置
        layers.append(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False))
        # print(layers)
        # 增加网络的非线性——激活函数nn.ReLU(True)  在卷积层（或BN层）之后，池化层之前，添加激活函数
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            # 构建卷积层
            layers.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=False))
            # 加快收敛速度一一批标准化层nn.BatchNorm2d()  输入通道为64的BN层 与卷积层输出通道数64对应
            # eps为保证数值稳定性（分母不能趋近或取0）,给分母加上的值。默认为1e-4
            # momentum： 动态均值和动态方差所使用的动量。默认为0.1
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            # 增加网络的非线性——激活函数nn.ReLU(True)  在卷积层（或BN层）之后，池化层之前，添加激活函数
            layers.append(nn.ReLU(inplace=True))
        # 构建卷积层
        layers.append(
            nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
                      bias=False))
        # 利用nn.Sequential()按顺序构建网络
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()  # 调用初始化权重函数

    # 定义自己的前向传播函数
    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y - out

    def _initialize_weights(self):
        for m in self.modules():
            # 使用isinstance来判断m属于什么类型【卷积操作】
            if isinstance(m, nn.Conv2d):
                # 正交初始化（Orthogonal Initialization）主要用以解决深度网络下的梯度消失、梯度爆炸问题
                init.orthogonal_(m.weight)
                # print('init weight')
                if m.bias is not None:
                    # init.constant_常数初始化
                    init.constant_(m.bias, 0)
            # 使用isinstance来判断m属于什么类型【批量归一化操作】
            elif isinstance(m, nn.BatchNorm2d):
                # init.constant_常数初始化
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


# 定义损失函数类
class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """

    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    # MSELoss  计算input和target之差的平方
    # reduce(bool)- 返回值是否为标量，默认为True size_average(bool)- 当reduce=True时有效。为True时，返回的loss为平均值；为False时，返回的各样本的loss之和。
    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == '__main__':
    # model selection
    print('===> Building model')
    model = DnCNN()
    # print(model)

    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        # 加载模型
        model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
    # 训练模型时会在前面加上
    model.train()
    # criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
    criterion = sum_squared_error()
    if cuda:
        model = model.cuda()
        # device_ids = [0]
        # model = nn.DataParallel(model, device_ids=device_ids).cuda()
        # criterion = criterion.cuda()
    # Optimizer  采用Adam算法优化，模型参数  学习率
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # milestones为一个数组，如 [50,70]. gamma为0.1 倍数。
    # 如果learning rate开始为0.01 ，则当epoch为50时变为0.001，epoch 为70 时变为0.0001。当last_epoch=-1,设定为初始lr
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates
    loss_list = []
    for epoch in range(initial_epoch, n_epoch):

        scheduler.step(epoch)  # step to the learning rate in this epcoh
        xs = dg.datagenerator(data_dir=args.train_data)  # 调用数据生成器函数
        xs = xs.astype('float32') / 255.0  # 对数据进行处理，位于【0 1】

        noise = dg.datagenerator(data_dir=args.noise_data)
        noise = noise.astype('float32') / 255.0

        # torch.from_numpy将numpy.ndarray 转换为pytorch的 Tensor。  transpose多维数组转置
        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
        noise = torch.from_numpy(noise.transpose((0, 3, 1, 2)))

        # 加噪声函数
        DDataset = DenoisingDataset(xs, noise, sigma)
        # dataset：（数据类型 dataset）
        # num_workers：工作者数量，默认是0。使用多少个子进程来导入数据
        # drop_last：丢弃最后数据，默认为False。设置了 batch_size 的数目后，最后一批数据未必是设置的数目，有可能会小些。这时你是否需要丢弃这批数据。
        # shuffle洗牌。默认设置为False。在每次迭代训练时是否将数据洗牌，默认设置是False。将输入数据的顺序打乱，是为了使数据更有独立性，但如果数据是有序列特征的，就不要设置成True了。
        DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0  # 初始化
        start_time = time.time()  # time.time() 返回当前时间的时间戳

        for n_count, batch_yx in enumerate(DLoader):  # enumerate() 函数用于将一个可遍历的数据对象
            optimizer.zero_grad()  # optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
            if cuda:
                batch_x, batch_y = batch_yx[1].cuda(), batch_yx[0].cuda()
            loss = criterion(model(batch_y), batch_x)  # 计算损失值
            epoch_loss += loss.item()  # 对损失值求和
            loss.backward()  # 反向传播
            optimizer.step()  # adam优化
            if n_count % 10 == 0:
                print('%4d %4d / %4d loss = %2.4f' % (
                epoch + 1, n_count, xs.size(0) // batch_size, loss.item() / batch_size))
        elapsed_time = time.time() - start_time

        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / n_count, elapsed_time))
        loss_value = str(epoch_loss / n_count)
        loss_list.append(loss_value)
        # numpy.savetxt(fname,X):第一个参数为文件名，第二个参数为需要存的数组（一维或者二维）第三个参数是保存的数据格式
        # hstack 和 vstack这两个函数分别用于在水平方向和竖直方向增加数据
        # np.savetxt('_202212train_result.txt', np.hstack((epoch + 1, epoch_loss / n_count, elapsed_time)), fmt='%2.4f')
        # torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        # 保存模型
        torch.save(model, os.path.join(save_dir, '202212model_%03d.pth' % (epoch + 1)))

    filename = save_dir + '_loss.txt'
    f = open(filename, 'w')  # 201809071117tcw
    for line in loss_list:  # 201809071117tcw
        f.write(line + '\n')  # 2018090711117tcw
    f.close()


