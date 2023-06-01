import torch
import torch.nn as nn
import torch.nn.init as init
from modelparts_t1 import _conv_,  _ResBlock_CBAM_

class DsCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DsCNN, self).__init__()
        kernel_size = 3  # 卷积核的大小  3*3
        padding = 1  # padding表示的是在图片周围填充0的多少，padding=0表示不填充，padding=1四周都填充1维
        features = 64
        layers = []
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True))
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True))
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True))
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True))
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True))
        self.conv13 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True))
        self.conv14 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True))
        self.conv15 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True))
        self.conv16 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95), nn.ReLU(inplace=True))
        self._rc_ = _ResBlock_CBAM_(in_channels=n_channels, kernel_size=3, stride=1, dilation=1, bias=True)
        self.conv17 = nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size,
                                padding=padding,
                                bias=False)
        self._initialize_weights()  # 调用初始化权重函数

    def forward(self, x):
        input = x
        out=self.conv1(x)
        out=self.conv2(out)
        out=self.conv3(out)
        out=self.conv4(out)
        out=self.conv5(out)
        out=self.conv6(out)
        out=self.conv7(out)
        out=self.conv8(out)
        out=self.conv9(out)
        out=self.conv10(out)
        out=self.conv11(out)
        out=self.conv12(out)
        out=self.conv13(out)
        out=self.conv14(out)
        out=self.conv15(out)
        out=self.conv16(out)
        out=self._rc_(out)
        out=self.conv17(out)
        return input - out

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
