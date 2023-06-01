import os


def generate(dir, label):
    '''
    dir：输入的文件路径，本例是E:/c/images
    label:要生成的标签
    '''

    # 返回dir中的所有文件，本例中是001.bad_apple和002.good_apple
    files = os.listdir(dir)
    # 文件夹中的数据是按序排好的，否则可以先排序
    # files.sort()

    # 要写入的文件名，若没有则生成一个新的
    listText = open('D:\CNN_Denoised\BM3D-Denoise-master\image_class_labels.txt', 'a')

    # 序号i
    i = 1
    for file in files:
        # 返回路径和文件名
        fileType = os.path.split(file)
        # print(file)
        # print(fileType)
        # 图片数据集中可能会有一个readme是.txt格式的，忽略它
        # readme文件也可能是其他格式的文件，根据修改即可
        # 若没有readme即可删掉
        if fileType[1] == '.txt':
            # print(fileType[1])
            continue
        # 生成序号和标签
        # i和label必须是同类型的
        name = str(i) + ' ' + str(int(label)) + '\n'
        # 生成文件名和标签
        # name = file + ' ' + str(int(label)) +'\n'
        listText.write(name)
        i += 1

    listText.close()


#图片数据集路径
out_path = r'D:/CNN_Denoised/BM3D-Denoise-master/noise/'

#标签i
i = 1
#获取到该文件夹下的所有子文件夹，即每一类的图片文件夹
folderlist = os.listdir(out_path)
#遍历每一类中的每一张图片
for folder in folderlist:
	#生成标签
    generate(os.path.join(out_path, folder),i)
    i += 1
