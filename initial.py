import os
import cv2
import numpy as np

# *****该模块主要用来初始化训练集和验证集，包括图像处理以及特征值的向量化*****

# 读取图片的目录
image_dir = 'img/'
# 保存数组的文件
array_file_train = 'label/'
array_file_val = 'label_val/'
# 设置阈值
thresh = 90
# 设置resize长宽高
width_r = 120
height_r = 90
# 设置裁剪后的resize长宽高
width_rr = 70
height_rr = 62


# 读取img目录下的图片转化为数组，保存到label中
def img_to_label():
    filenames = os.listdir(image_dir)  # 获取文件夹下所有的图片的命名
    for filename in filenames:
        # 读取图片
        img = cv2.imread(image_dir + filename, cv2.IMREAD_GRAYSCALE)  # 读取灰度图
        resized_img = cv2.resize(img, (width_r, height_r))  # 按比例缩放
        resized_img = resized_img[int(height_r * 0.23):int(height_r * 0.92),
                      int(width_r * 0.15):int(width_r * 0.74)]  # 按照比例裁剪
        # 进行二值化操作
        new_img = resized_img  # 重新命名
        height, width = new_img.shape[0:2]  # 得到图像的高度和宽度
        # 遍历像素点
        for row in range(height):
            for col in range(width):
                gray = new_img[row, col]  # 获取到灰度值
                if gray > thresh:  # 大于阈值255
                    new_img[row, col] = 255
                else:  # 否则0
                    new_img[row, col] = 0
        print("filename", filename)  # 打印正在处理的图片名
        image_arr = np.array(new_img).reshape(width_rr * height_rr)  # 将其拼接为一维数组
        # 重新分配类名 例如：A对应001 B对应002
        if int(filename[4:6]) - 10 < 10:  # 判断是否个位数
            class_name = "00" + str(int(filename[4:6]) - 10)  # 个位数的名字在前面加两个0，统一命名规则
        else:
            class_name = "0" + str(int(filename[4:6]) - 10)  # 两位数的名字在前面加一个0，统一命名规则
        num_name = filename[7:10]  # 同一类中的编号名
        save_name = class_name + "_" + num_name  # 保存的txt文件名
        print("save name", save_name)  # 打印保存的文件名
        ran = np.random.randint(0, 10)  # 随机数分配训练集或者验证集
        if ran < 8:  # 随机数是0-10，故小于8为训练集，否则为验证集
            np.savetxt(array_file_train + save_name + ".txt", image_arr, delimiter=",", fmt='%d')  # 保存到训练集中
        else:
            np.savetxt(array_file_val + save_name + ".txt", image_arr, delimiter=",", fmt='%d')  # 保存到验证集中


if __name__ == '__main__':
    img_to_label()  # 运行该函数
