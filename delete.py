import os

def del_files(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(".tif"): # 特定图片格式
                os.remove(os.path.join(root, name))
                print("Delete File: " + os.path.join(root, name))

if __name__ == "__main__":
    path = r"E:\train\\"
    del_files(path)
# 这里是删除'D:\shijue\animal\picture'路径下所有.png格式的图片，修改“特定图片格式”处，可删除任意格式的图片
