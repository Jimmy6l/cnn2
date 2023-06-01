import torch
import torch.nn as nn
import torch.nn.init as init
from modelparts_t2 import _conv_, _ResBlock_CBAM_, _context_block_

# 膨胀后的卷积核尺寸 = 膨胀系数 × (原始卷积核尺寸-1）+ 1
class DCNND(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DCNND, self).__init__()
        kernel_size = 3  # 卷积核的大小  3*3
        padding = 1  # padding表示的是在图片周围填充0的多少，padding=0表示不填充，padding=1四周都填充1维
        features = 64
        layers = []
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=2,
                      bias=False, dilation=2), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=3,
                      bias=False, dilation=3), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=5,
                      bias=False, dilation=4), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=3,
                      bias=False, dilation=3), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95),
            nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=2,
                      bias=False, dilation=2), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95),
            nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True))
        self._initialize_weights()  # 调用初始化权重函数

    def forward(self, x):
        input = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)

        return  out

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
