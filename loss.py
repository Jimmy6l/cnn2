import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        # Inheritance
        super(Loss, self).__init__()

        # Initialize Loss Weight
        self._lambda_tv_ = 2e-3

        # Create Loss Instance
        # self._loss_function_ = nn.MSELoss()
        self._mse_loss_ = sum_squared_error()
        self._tv_loss_ = TVLoss()

    def forward(self, inputs, targets):
        # Pixel Loss
        dg_loss = self._mse_loss_(inputs, targets)

        # TV Loss
        tv_loss = self._tv_loss_(inputs)

        return dg_loss + self._lambda_tv_ * tv_loss

class sum_squared_error(nn.Module):  # PyTorch 0.4.1
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


class TVLoss(nn.Module):
    def __init__(self):
        # Inheritance
        super(TVLoss, self).__init__()

    def forward(self, x):
        # Initialize Variables
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]

        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])

        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()

        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
