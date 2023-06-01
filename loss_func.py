#  __author__ = 'czx'
# coding=utf-8
import numpy as np
# from numpy import *
# import matplotlib.pyplot as plt
#
# def relu(x):
#     y = x.copy()
#     y[y < 0] = 0
#     return y
#
# x = np.arange(-10, 10, 0.01)
# plt.tick_params(labelsize=14)  # 刻度字体大小14
# y_relu = relu(x)
#
# plt.plot(x, y_relu, 'b', linewidth=2.5, label=u'ReLU')
# plt.grid(True,linestyle=':')
# plt.legend(loc='upper left',fontsize=16, frameon=False)  # 图例字体大小16
# plt.tight_layout()  # 去除边缘空白
# plt.savefig("D:\\Code_Data\\0Proposed_DnCNN\\relu.jpeg", dpi=600, format="jpeg")
# #savefig要写在show前面,不然保存的就是空白图片
# plt.show()
#  __author__ = 'czx'
# coding=utf-8
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist


def leakyrelu(x):
    y = x.copy()
    for i in range(y.shape[0]):
        if y[i] < 0:
            y[i] = 0.2 * y[i]
    return y

x = np.arange(-10, 10, 0.01)
plt.tick_params(labelsize=14)  # 刻度字体大小14
y_relu = leakyrelu(x)

plt.plot(x, y_relu, 'b', linewidth=2.5, label=u'LeakyReLU')
plt.grid(True,linestyle=':')
plt.legend(loc='upper left', fontsize=16, frameon=False)  # 图例字体大小16
plt.tight_layout()  # 去除边缘空白
plt.savefig("D:\\Code_Data\\0Proposed_DnCNN\\leakyrelu.jpeg", dpi=600, format="jpeg")
# savefig要写在show前面,不然保存的就是空白图片
plt.show()

