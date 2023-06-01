import cv2
import os

def cv2_imread():
    # 图片路径，相对路径
    # image_path = r"D:\CNN_Denoised\data\vedio\3\4.jpg"
    # 读取图片,格式为BGR
    # image = cv2.imread(image_path,0)
    # 显示图片形状
    # print("image_shape: ", image.shape)
    # 显示图片
    # cv2.imshow('img', image)
    # cropImg = image[110:687, 96:891]
    # print("image_shape: ", cropImg.shape)
    # cv2.imwrite(r'D:\CNN_Denoised\data\vedio\res.jpg',cropImg)
    # cv2.imshow('img2', cropImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    image_path = r"E:\ultrasound dataset\test_res\\"
    new_path = r"E:\data\clean\\"
    for filename in os.listdir(image_path):
        print(filename)
        image = cv2.imread(image_path + filename,0)
        # cropImg = image[110:687, 96:891]
        cv2.imwrite(new_path +'sigma0.4_'+ filename,image)




if __name__ == '__main__':
    cv2_imread()