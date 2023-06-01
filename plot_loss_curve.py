import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

plt.rcParams['font.family'] = 'FangSong'  # 设置字体为仿宋
plt.rcParams['font.size'] = 10  # 设置字体的大小为10


def ReadTxtName(rootdir):
    loss = []
    with open(rootdir, 'r') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            line = line.strip('\n')
            loss.append(float(line))
        return loss


if __name__ == '__main__':
    rootdir = r'D:\Code_Data\0Proposed_DnCNN\results\DsCNN_3_sigma28_loss.txt'
    losslist1 = ReadTxtName(rootdir)
    print(losslist1)
    ax = plt.subplots()

    x = range(0, 300)  # 100
    y1 = losslist1
    # y2 = lineslist2
    # y3 = lineslist3
    # y4 = lineslist4
    plt.title('Result Analysis')
    plt.plot(x, y1, color='black', label='training log loss')  # 四条曲线
    # plt.plot(x, y2, color='blue', label='Net2')
    # plt.plot(x, y3, color='red', label='Net3')
    # plt.plot(x, y4, color='green', label='improveNet')
    plt.xlim(10, 100)
    plt.ylim(40,200)
    plt.legend()  # 显示图例

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
