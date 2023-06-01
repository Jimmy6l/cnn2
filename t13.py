import cv2
import os
#截取区域
def cv2_imread():
    # # 图片路径，相对路径
    # image_path = r"D:\Code_Data\data\Philips_data\4.jpg"
    # # 读取图片,格式为BGR
    # image = cv2.imread(image_path,0)
    # # 显示图片形状
    # print("image_shape: ", image.shape)
    # # 显示图片
    # cv2.imshow('img', image)
    # # H，W
    # cropImg = image[70:720, 145:875]
    # print("image_shape: ", cropImg.shape)
    # # cv2.imwrite(r'D:\CNN_Denoised\data\vedio\res.jpg',cropImg)
    # cv2.imshow('img2', cropImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    image_path = "D:\Code_Data\data\Philips_data\\"
    new_path = "D:\\CNN_Denoised\\data\\vedio\\r\\"
    for filename in os.listdir(image_path):
        print(filename)
        image = cv2.imread(image_path + filename,0)
        cropImg = image[70:720, 145:875]
        # cv2.imwrite(new_path +'r3_'+ filename,image)
        cv2.imwrite(image_path + filename,cropImg)




if __name__ == '__main__':
    cv2_imread()