# -*- coding: utf-8 -*-

# PyTorch 0.4.1, https://pytorch.org/docs/stable/index.html

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

# run this to train the model

# =============================================================================
# For batch normalization layer, momentum should be a value from [0.1, 1] rather than the default 0.1.
# The Gaussian noise output helps to stablize the batch normalization, thus a large momentum (e.g., 0.95) is preferred.
# =============================================================================

import argparse
import datetime
import glob
import os
import re
import time

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from torch.nn.functional import pad
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import data_generator1 as dg
from data_generator1 import DenoisingDataset
from padding_same_conv import Conv2dSame
# Params
# 创建一个解析器
# 使用 argparse 的第一步是创建一个 ArgumentParser 对象
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
# 给一个 ArgumentParser 添加程序参数信息是通过调用 add_argument() 方法完成的。
parser.add_argument('--model', default='S_DnCNN', type=str, help='choose a type of model')
# 批量大小  整型   默认大小128
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
# 训练数据   字符串  默认 data/Train400  路径
parser.add_argument('--train_data', default='D:\\CNN_Denoised\\OBNLM\\2021test2laplacian', type=str,
                    help='path of train data')
parser.add_argument('--noise_data', default='D:\\CNN_Denoised\\OBNLM\\2021test2', type=str, help='path of train data')
# 噪声水平  整型  默认25
parser.add_argument('--sigma', default=24, type=int, help='noise level')
# epoch 整型  默认180
parser.add_argument('--epoch', default=20, type=int, help='number of train epoches')
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
save_dir = os.path.join('kd_models', args.model + '_' + 'sigma' + str(sigma))

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


# 定义构造函数
# 构建网络最开始写一个class，然后def _init_（输入的量），然后super（DnCNN，self）.__init__()这三句是程式化的存在，
# 初始化
# 修改nn.Conv2d为Conv2dSame
class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3  # 卷积核的大小  3*3
        padding = 1  # padding表示的是在图片周围填充0的多少，padding=0表示不填充，padding=1四周都填充1维
        layers = []
        # 四个参数 输入的通道  输出的通道  卷积核大小  padding
        # 构建一个输入通道为channels，输出通道为64，卷积核大小为3*3,四周进行1个像素点的零填充的conv1层  #bias如果bias=True，添加偏置
        layers.append(
            Conv2dSame(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False))
        # 增加网络的非线性——激活函数nn.ReLU(True)  在卷积层（或BN层）之后，池化层之前，添加激活函数
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            # 构建卷积层
            layers.append(
                Conv2dSame(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=False))
            # 加快收敛速度一一批标准化层nn.BatchNorm2d()  输入通道为64的BN层 与卷积层输出通道数64对应
            # eps为保证数值稳定性（分母不能趋近或取0）,给分母加上的值。默认为1e-4
            # momentum： 动态均值和动态方差所使用的动量。默认为0.1
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            # 增加网络的非线性——激活函数nn.ReLU(True)  在卷积层（或BN层）之后，池化层之前，添加激活函数
            layers.append(nn.ReLU(inplace=True))
        layers.append(SELayer(channel=n_channels))
        # 构建卷积层
        layers.append(
            Conv2dSame(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
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

# 定义SE注意力模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

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

# 定义student模型
class S_DnCNN(nn.Module):
    def __init__(self, depth=9, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(S_DnCNN, self).__init__()
        kernel_size = 3  # 卷积核的大小  3*3
        padding = 1  # padding表示的是在图片周围填充0的多少，padding=0表示不填充，padding=1四周都填充1维
        layers = []
        # 四个参数 输入的通道  输出的通道  卷积核大小  padding
        # 构建一个输入通道为channels，输出通道为64，卷积核大小为3*3,四周进行1个像素点的零填充的conv1层  #bias如果bias=True，添加偏置
        layers.append(
            Conv2dSame(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False))
        # 增加网络的非线性——激活函数nn.ReLU(True)  在卷积层（或BN层）之后，池化层之前，添加激活函数
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            # 构建卷积层
            layers.append(
                Conv2dSame(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=False))
            # 加快收敛速度一一批标准化层nn.BatchNorm2d()  输入通道为64的BN层 与卷积层输出通道数64对应
            # eps为保证数值稳定性（分母不能趋近或取0）,给分母加上的值。默认为1e-4
            # momentum： 动态均值和动态方差所使用的动量。默认为0.1
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            # 增加网络的非线性——激活函数nn.ReLU(True)  在卷积层（或BN层）之后，池化层之前，添加激活函数
            layers.append(nn.ReLU(inplace=True))
        layers.append(SELayer(channel=n_channels))
        # 构建卷积层
        layers.append(
            Conv2dSame(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
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

if __name__ == '__main__':
    # 训练学生模型
    model = S_DnCNN()
    model.train()
    criterion = sum_squared_error()
    if cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates
    loss_list = []
    for epoch in range(n_epoch):
        scheduler.step(epoch)  # step to the learning rate in this epcoh
        xs = dg.datagenerator(data_dir=args.train_data)  # 调用数据生成器函数
        xs = xs.astype('float32') / 255.0  # 对数据进行处理，位于【0 1】

        noise = dg.datagenerator(data_dir=args.noise_data)
        noise = noise.astype('float32') / 255.0

        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
        noise = torch.from_numpy(noise.transpose((0, 3, 1, 2)))

        DDataset = DenoisingDataset(xs, noise, sigma)

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

        # 保存模型
        if epoch % 9 == 0:
            torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))

    # load 教师模型
    teach_model = DnCNN()
    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        # 加载模型
        # model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
        teach_model = torch.load('models/DnCNN_sigma22/model_180.pth')
        print('teacher:', teach_model)

    teach_model.eval()

    # 准备新的学生模型
    student_model = S_DnCNN()
    if cuda:
        student_model = student_model.cuda()
    student_model.train()

    # 蒸馏温度
    temp = 7

    # 损失
    loss = nn.CrossEntropyLoss()

    # optim
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates
    loss_list = []

    epochs = 20
    for epoch in range(epochs):

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

            # 教师模型预测
            with torch.no_grad():
                teacher_preds = teach_model(batch_y)

            # 学生模型预测
            student_preds = student_model(batch_y)

            loss = criterion(teacher_preds, student_preds)  # 计算损失值
            epoch_loss += loss.item()  # 对损失值求和
            loss.backward()  # 反向传播
            optimizer.step()  # adam优化
            if n_count % 10 == 0:
                print('%4d %4d / %4d loss = %2.4f' % (
                epoch + 1, n_count, xs.size(0) // batch_size, loss.item() / batch_size))
        elapsed_time = time.time() - start_time

        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / n_count, elapsed_time))
        loss_value=str(epoch_loss / n_count)
        loss_list.append(loss_value)

        # 保存模型
        # torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))
        if epoch % 9 == 0:
            torch.save(student_preds, os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))

    filename = save_dir + '_loss.txt'
    f = open(filename,'w') #201809071117tcw
    for line in loss_list:  #201809071117tcw
        f.write(line+'\n') #2018090711117tcw
    f.close()




