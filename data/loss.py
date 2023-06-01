import numpy as np
import matplotlib.pyplot as plt
batch_size=8
# 显示loss曲线
loss_lines = np.loadtxt('D:\\CNN_Denoised\\DnCNN_pytorch\\models\\DnCNN_sigma20_loss.txt')
# 前面除以batch_size会导致数值太小了不易观察
train_line = loss_lines[:] / batch_size
valida_line = train_line
x1 = range(len(train_line))
fig1 = plt.figure()
plt.plot(x1, train_line, x1, valida_line)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'valida'])
plt.savefig('loss_plot.png', bbox_inches='tight')
plt.tight_layout()