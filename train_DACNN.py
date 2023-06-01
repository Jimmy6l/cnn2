import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import data_generator2 as dg
from data_generator2 import DenoisingDataset
# import data_generator1 as dg
# from data_generator1 import DenoisingDataset
from loss_abl1 import Loss
from DACNN import DACNN
from skimage.measure import compare_psnr, compare_ssim # 计算图像的峰值信噪比（PSNR）  #计算两幅图像之间的平均结构相似性指数。


parser = argparse.ArgumentParser(description='PyTorch DACNN')
parser.add_argument('--model', default='DACNN_', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--train_data', default='E:\\0CNN\\DATASET\\clean', type=str,
                    help='path of train data')
parser.add_argument('--noise_data', default='E:\\0CNN\\DATASET\\noise', type=str,
                    help='path of train data')
parser.add_argument('--sigma', default=28, type=int, help='noise level')
parser.add_argument('--epoch', default=2, type=int, help='number of train epoches')
parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate for Adam')
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")

args = parser.parse_args()

batch_size = args.batch_size
cuda = torch.cuda.is_available()
n_epoch = args.epoch
sigma = args.sigma
# os.path.join()：  将多个路径组合后返回args.model = DNCNN  str(sigma)=25
# 组合之后的路径为models/DNCNN_sigma25
save_dir = os.path.join('models', args.model + '_' + 'sigma' + str(sigma))

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

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

if __name__ == '__main__':
    # model selection
    print('===> Building model')
    model = DACNN()

    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        # 加载模型
        model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
    # 训练模型时会在前面加上
    model.train()
    # criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
    criterion = Loss()
    # criterion = sum_squared_error()
    if cuda:
        model = model.cuda()
        # device_ids = [0]
        # model = nn.DataParallel(model, device_ids=device_ids).cuda()
        # criterion = criterion.cuda()
    # Optimizer  采用Adam算法优化，模型参数  学习率
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # milestones为一个数组，如 [50,70]. gamma为0.1 倍数。
    # 如果learning rate开始为0.01 ，则当epoch为50时变为0.001，epoch 为70 时变为0.0001。当last_epoch=-1,设定为初始lr
    # scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates
    loss_list = []
    psnr_list = []
    for epoch in range(initial_epoch, n_epoch):
        if epoch <= args.milestone:
            current_lr = args.lr
        if epoch > 30 and  epoch <=60:
            current_lr  =  args.lr/10.
        if epoch > 60  and epoch <=90:
            current_lr = args.lr/100.
        if epoch >90 and epoch <=180:
            current_lr = args.lr/1000.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)

        # scheduler.step(epoch)  # step to the learning rate in this epcoh
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
            # Get Loss
            loss = criterion(model(batch_y), batch_x)  # 计算损失值
            # loss = criterion(model(batch_y), batch_x)  # 计算损失值
            epoch_loss += loss.item()  # 对损失值求和
            loss.backward()  # 反向传播
            optimizer.step()  # adam优化
            if n_count % 10 == 0:
                print('%4d %4d / %4d loss = %2.4f' % (
                epoch + 1, n_count, xs.size(0) // batch_size, loss.item() / batch_size))
            model.eval()
            out_train = torch.clamp(model(batch_y), 0., 1.)
            psnr_train = compare_psnr(batch_y, batch_x)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                  (epoch + 1, n_count + 1, len(DLoader), loss.item(),psnr_train))
        elapsed_time = time.time() - start_time

        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / n_count, elapsed_time))
        loss_value = str(epoch_loss / n_count)
        loss_list.append(loss_value)
        # 保存模型
        np.savetxt(os.path.join(save_dir,'train_result.txt'), np.hstack((epoch + 1, epoch_loss / n_count, elapsed_time)), fmt='%2.4f')
        torch.save(model, os.path.join(save_dir, '202212model_%03d.pth' % (epoch + 1)))

    filename = save_dir + '_loss.txt'
    f = open(filename, 'w')  # 201809071117tcw
    for line in loss_list:  # 201809071117tcw
        f.write(line + '\n')  # 2018090711117tcw
    f.close()


