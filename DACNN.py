import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class _conv_(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, bias):
        # Inheritance
        super(_conv_, self).__init__()

        # Create Layer Instance
        self._conv_ = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(dilation * (kernel_size - 1)) // 2,
            dilation=dilation,
            bias=bias
        )

    def forward(self, x):
        out = self._conv_(x)

        return out


class _conv_block_(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, bias):
        # Inheritance
        super(_conv_block_, self).__init__()

        # Create Layer Instance
        self._conv_in_ = _conv_(in_channels, out_channels, kernel_size, stride, dilation, bias)

    def forward(self, x):
        out = self._conv_in_(x)
        out = F.leaky_relu(out, 0.2, True)

        return out


class _channel_attention_module_(nn.Module):
    def __init__(self, in_channels, stride, dilation, bias):
        # Inheritance
        super(_channel_attention_module_, self).__init__()

        # Create Layer Instance
        self._aap_ = nn.AdaptiveAvgPool2d(1)
        self._amp_ = nn.AdaptiveMaxPool2d(1)
        self._conv_ = nn.Sequential(
            _conv_block_(in_channels, in_channels // 4, 1, stride, dilation, bias),
            _conv_(in_channels // 4, in_channels, 1, stride, dilation, bias)
        )

    def forward(self, x):
        out = self._conv_(self._aap_(x)) + self._conv_(self._amp_(x))
        out = F.sigmoid(out)

        return out


class _spatial_attention_module_(nn.Module):
    def __init__(self, in_channels, stride, dilation, bias):
        # Inheritance
        super(_spatial_attention_module_, self).__init__()

        # Create Layer Instance
        self._bottleneck_ = _conv_(2, 1, 7, stride, dilation, bias)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self._bottleneck_(out)
        out = F.sigmoid(out)

        return out


class DACNN(nn.Module):
    def __init__(self, depth=10, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DACNN, self).__init__()
        kernel_size = 3  # 卷积核的大小  3*3
        padding = 1  # padding表示的是在图片周围填充0的多少，padding=0表示不填充，padding=1四周都填充1维
        features = 64
        layers = []
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False), nn.ReLU(inplace=True))
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=False)
        self._cam_2 = _channel_attention_module_(in_channels=n_channels, stride=1, dilation=1, bias=True)
        self._sam_2 = _spatial_attention_module_(in_channels=n_channels, stride=1, dilation=1, bias=True)
        self._conv_out_2 = _conv_(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, stride=1,
                                 dilation=1, bias=True)
        self._bn_2 = nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95)

        self.conv3 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                               padding=padding,
                               bias=False)
        self._cam_3 = _channel_attention_module_(in_channels=n_channels, stride=1, dilation=1, bias=True)
        self._sam_3 = _spatial_attention_module_(in_channels=n_channels, stride=1, dilation=1, bias=True)
        self._conv_out_3 = _conv_(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, stride=1,
                                  dilation=1, bias=True)
        self._bn_3 = nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95)
        self.conv4 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                               padding=padding,
                               bias=False)
        self._cam_4 = _channel_attention_module_(in_channels=n_channels, stride=1, dilation=1, bias=True)
        self._sam_4 = _spatial_attention_module_(in_channels=n_channels, stride=1, dilation=1, bias=True)
        self._conv_out_4 = _conv_(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, stride=1,
                                  dilation=1, bias=True)
        self._bn_4 = nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95)
        self.conv5 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                               padding=padding,
                               bias=False)
        self._cam_5 = _channel_attention_module_(in_channels=n_channels, stride=1, dilation=1, bias=True)
        self._sam_5 = _spatial_attention_module_(in_channels=n_channels, stride=1, dilation=1, bias=True)
        self._conv_out_5 = _conv_(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, stride=1,
                                  dilation=1, bias=True)
        self._bn_5 = nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95)
        self.conv6 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                               padding=padding,
                               bias=False)
        self._cam_6 = _channel_attention_module_(in_channels=n_channels, stride=1, dilation=1, bias=True)
        self._sam_6 = _spatial_attention_module_(in_channels=n_channels, stride=1, dilation=1, bias=True)
        self._conv_out_6 = _conv_(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, stride=1,
                                  dilation=1, bias=True)
        self._bn_6 = nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95)
        self.conv7 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                               padding=padding,
                               bias=False)
        self._cam_7 = _channel_attention_module_(in_channels=n_channels, stride=1, dilation=1, bias=True)
        self._sam_7 = _spatial_attention_module_(in_channels=n_channels, stride=1, dilation=1, bias=True)
        self._conv_out_7 = _conv_(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, stride=1,
                                  dilation=1, bias=True)
        self._bn_7 = nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95)
        self.conv8 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                               padding=padding,
                               bias=False)
        self._cam_8 = _channel_attention_module_(in_channels=n_channels, stride=1, dilation=1, bias=True)
        self._sam_8 = _spatial_attention_module_(in_channels=n_channels, stride=1, dilation=1, bias=True)
        self._conv_out_8 = _conv_(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, stride=1,
                                  dilation=1, bias=True)
        self._bn_8 = nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95)
        self.conv9 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                               padding=padding,
                               bias=False)
        self._cam_9 = _channel_attention_module_(in_channels=n_channels, stride=1, dilation=1, bias=True)
        self._sam_9 = _spatial_attention_module_(in_channels=n_channels, stride=1, dilation=1, bias=True)
        self._conv_out_9 = _conv_(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, stride=1,
                                  dilation=1, bias=True)
        self._bn_9 = nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95)
        self.conv10 = nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
                      bias=False)
        self._initialize_weights()  # 调用初始化权重函数

    def forward(self, x):
        input = x
        out = self.conv1(x)
        x1=out
        out = self.conv2(out)
        out = out * self._cam_2(out)
        out = out * self._sam_2(out)
        out = self._conv_out_2(out + x1)
        out=self._bn_2(out)
        x2 = out
        out = self.conv3(out)
        out = out * self._cam_3(out)
        out = out * self._sam_3(out)
        out = self._conv_out_3(out + x2)
        out=self._bn_3(out)
        x3 = out
        out = self.conv4(out)
        out = out * self._cam_4(out)
        out = out * self._sam_4(out)
        out = self._conv_out_4(out + x3)
        out = self._bn_4(out)
        x4 = out
        out = self.conv5(out)
        out = out * self._cam_5(out)
        out = out * self._sam_5(out)
        out = self._conv_out_5(out + x4)
        out = self._bn_5(out)
        x5 = out
        out = self.conv6(out)
        out = out * self._cam_6(out)
        out = out * self._sam_6(out)
        out = self._conv_out_6(out + x5)
        out = self._bn_6(out)
        x6 = out
        out = self.conv7(out)
        out = out * self._cam_7(out)
        out = out * self._sam_7(out)
        out = self._conv_out_7(out + x6)
        out = self._bn_7(out)
        x7 = out
        out = self.conv8(out)
        out = out * self._cam_8(out)
        out = out * self._sam_8(out)
        out = self._conv_out_8(out + x7)
        out = self._bn_8(out)
        x8 = out
        out = self.conv9(out)
        out = out * self._cam_9(out)
        out = out * self._sam_9(out)
        out = self._conv_out_9(out + x8)
        out = self._bn_9(out)
        out = self.conv10(out)
        out = F.sigmoid(out)
        return out

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
