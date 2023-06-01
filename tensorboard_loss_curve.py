import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


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

    writer = SummaryWriter(log_dir="summary_pic")  # 第二步，确定保存的路径，会保存一个文件夹，而非文件
    rootdir = r'D:\Code_Data\0Proposed_DnCNN\results\DsCNN_3_sigma28_loss.txt'
    losslist1 = ReadTxtName(rootdir)
    for epoch in range(150):
        writer.add_scalar("loss", losslist1[epoch], epoch)  # 第三步，绘图
    writer.close()  # 第4步，写入关闭
# tensorboard --logdir "./summary_pic"

    # loss_list = [0]
    # psnr_list = []
    # # loss_value=''
    # save_dir='D:\\Code_Data\\0Proposed_DnCNN\\'
    # filename =save_dir  + '_loss_epoch.txt'
    # f = open(filename, 'w')  # 201809071117tcw
    # for epoch in range(6):
    #     f.write(str(loss_list[epoch]) + '\n')
    #     for n_count in range(5):
    #         loss_value = str(n_count)
    #         loss_list.append(loss_value)
