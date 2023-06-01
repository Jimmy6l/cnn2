import argparse
import os, time, datetime
# import PIL.Image as Image
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from skimage.measure import compare_psnr, compare_ssim # 计算图像的峰值信噪比（PSNR）  #计算两幅图像之间的平均结构相似性指数。
from skimage.io import imread, imsave  # 在python中，图像处理主要采用的库：skimage, opencv-python, Pillow (PIL)。 这三个库均提供了图像读取的方法。
from models_t1 import DsCNN

# 创建解析器
def parse_args():
    parser = argparse.ArgumentParser()
    # 增加参数
    # 测试集  data/Test
    parser.add_argument('--set_dir', default='data/Test', type=str, help='directory of test dataset')
    # 测试集名字  set68 set12 , 'Set12'
    parser.add_argument('--set_names', default=['Set14'], help='directory of test dataset')
    # 噪声水平  默认25
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    # 模型位置  models/DnCNN_sigma25
    parser.add_argument('--model_dir', default=os.path.join('models', 'DnCNN_sigma27'), help='directory of the model')
    # 模型名字 默认model_001.pth
    parser.add_argument('--model_name', default='202212model_300.pth', type=str, help='the model name')
    # 结果位置  results
    parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
    # 保存结果 1保存 0不保存
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


# 保存结果
def save_result(result, path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))


# 展示图片
def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


if __name__ == '__main__':
    # 调用解析器
    args = parse_args()

    model = DsCNN()
    # 如果路径下的模型不存在models/DnCNN_sigma25/model_001.pth
    if not os.path.exists(os.path.join(args.model_dir, args.model_name)):
        # 加载路径下的模型models/DnCNN_sigma25/model.pth
        model = torch.load(os.path.join(args.model_dir, 'model.pth'))
        # load weights into new model
        # 加载源代码中自带的训练模型
        log('load trained model on Train400 dataset by kai')
    else:
        # model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
        # 加载自己训练的模型
        model = torch.load(os.path.join(args.model_dir, args.model_name))
        log('load trained model')

#    params = model.state_dict()
#    print(params.values())
#    print(params.keys())
#
#    for key, value in params.items():
#        print(key)    # parameter name
#    print(params['dncnn.12.running_mean'])
#    print(model.state_dict())
    # 测试模型时会在前面加上
    model.eval()  # evaluation mode
#    model.train()
    # 判断GPU是否可用    可用的话，model放在GPU上测试
    if torch.cuda.is_available():
        model = model.cuda()
    # 如果存放结果文件夹不存在，创建文件夹
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    for set_cur in args.set_names:
        # 如果存放结果的文件夹不存在，创建Test/set12   Test/set68
        if not os.path.exists(os.path.join(args.result_dir, set_cur)):
            os.mkdir(os.path.join(args.result_dir, set_cur))
        psnrs = []  # 峰值信噪比
        ssims = []  # 结构相似性
        # args.set_dir：data/Test
        # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
        for im in os.listdir(os.path.join(args.set_dir, set_cur)):
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                # endswith() 方法用于判断字符串是否以指定后缀结尾，如果以指定后缀结尾返回True，否则返回False
                x = np.array(imread(os.path.join(args.set_dir, set_cur, im)), dtype=np.float32)/255.0
                # np.array将读入的图像转换为ndarray形式，且使像素值处于[0 1]
                np.random.seed(seed=0)  # for reproducibility 随机生成一个种子
                # seed( ) 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，则每次生成的随即数都相同
                # y = x + np.random.normal(0, args.sigma/255.0, x.shape)  # Add Gaussian noise without clipping
                y = x   # No Add Gaussian noise without clipping
                y = y.astype(np.float32) # dtype 用于查看数据类型   astype 用于转换数据类型
                # 将numpy矩阵y转化为torch张量y_，共享内存
                # np.reshape()和torch.view()效果一样，reshape（）操作nparray，view（）操作tensor
                y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])
                # 正确的测试时间的代码 torch.cuda.synchronize()
                torch.cuda.synchronize()
                start_time = time.time()
                y_ = y_.cuda()
                x_ = model(y_)  # inference
                x_ = x_.view(y.shape[0], y.shape[1])
                x_ = x_.cpu()
                x_ = x_.detach().numpy().astype(np.float32)
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                # 输出测试集名字 ：图片名  ：  运行时间
                print('%10s : %10s : %2.4f second' % (set_cur, im, elapsed_time))

                psnr_x_ = compare_psnr(x, x_)  # skimage.measure库里的compare_psnr方法
                ssim_x_ = compare_ssim(x, x_)  # skimage.measure库里的compare_ssim方法
                # 如果存放结果的文件夹存在
                if args.save_result:
                    # 分离文件名与扩展名
                    name, ext = os.path.splitext(im)
                    # 调用自定义函数show()展示图片
                    # numpy.hstack()函数是将数组沿水平方向堆叠起来
                    show(np.hstack((y, x_)))  # show the image
                    # 调用保存结果函数  参数（X_,路径results/set12/文件名_dncnn.后缀）
                    save_result(x_, path=os.path.join(args.result_dir, set_cur, name+'_dscnn'+ext))  # save the denoised image
                psnrs.append(psnr_x_) # 保存PSNR
                ssims.append(ssim_x_) # 保存SSIM
        # numpy.mean() 函数返回数组中元素的算术平均值
        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)
        # 如果文件夹存在
        if args.save_result:
            # 保存（psnr和ssims值，放在文件results.txt里面）
            save_result(np.hstack((psnrs, ssims)), path=os.path.join(args.result_dir, set_cur, 'results.txt'))
        # 打印日志，调用log函数
        log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))
