import torch
import torch.nn as nn
import torch.nn.functional as F

class _conv_(nn.Module) :
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, bias) :
        # Inheritance
        super(_conv_, self).__init__()

        # Create Layer Instance
        self._conv_ = nn.Conv2d(
                            in_channels = in_channels,
                            out_channels = out_channels,
                            kernel_size = kernel_size,
                            stride = stride,
                            padding = (dilation * (kernel_size - 1)) // 2 ,
                            dilation = dilation,
                            bias = bias
                            )

    def forward(self, x) :
        out = self._conv_(x)

        return out

class _conv_block_(nn.Module) :
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, bias) :
        # Inheritance
        super(_conv_block_, self).__init__()

        # Create Layer Instance
        self._conv_in_ =  _conv_(in_channels, out_channels, kernel_size, stride, dilation, bias)

    def forward(self, x) :
        out = self._conv_in_(x)
        out = F.leaky_relu(out, 0.2, True)

        return  out

class _channel_attention_module_(nn.Module) :
    def __init__(self, in_channels, stride, dilation, bias) :
        # Inheritance
        super(_channel_attention_module_, self).__init__()

       # Create Layer Instance
        self._aap_ = nn.AdaptiveAvgPool2d(1)
        self._amp_ = nn.AdaptiveMaxPool2d(1)
        self._conv_ = nn.Sequential(
                            _conv_block_(in_channels, in_channels // 4, 1, stride, dilation, bias),
                            _conv_(in_channels // 4, in_channels, 1, stride, dilation, bias)
                            )

    def forward(self, x) :
        out = self._conv_(self._aap_(x)) + self._conv_(self._amp_(x))
        out = F.sigmoid(out)

        return out

class _spatial_attention_module_(nn.Module) :
    def __init__(self, in_channels, stride, dilation, bias) :
        # Inheritance
        super(_spatial_attention_module_, self).__init__()

        # Create Layer Instance
        self._bottleneck_ = _conv_(2, 1, 7, stride, dilation, bias)

    def forward(self, x) :
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self._bottleneck_(out)
        out = F.sigmoid(out)

        return out

class _ResBlock_CBAM_(nn.Module) :
    def __init__(self, in_channels, kernel_size, stride, dilation, bias) :
        # Inheritance
        super(_ResBlock_CBAM_, self).__init__()

        # Create Layer Instance
        self._conv_in_ = _conv_(in_channels, in_channels, kernel_size, stride, dilation, bias)
        self._conv_out_ = _conv_(in_channels, in_channels, kernel_size, stride, dilation, bias)
        self._cam_ = _channel_attention_module_(in_channels, stride, dilation, bias)
        self._sam_ = _spatial_attention_module_(in_channels, stride, dilation, bias)

    def forward(self, x) :
        out = self._conv_in_(x)
        out = out * self._cam_(out)
        out = out * self._sam_(out)
        out = self._conv_out_(out + x)

        return out
