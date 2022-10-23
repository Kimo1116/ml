import numpy as np
import operator
from os import listdir
import cv2

# # 设置resize长宽高
# width_r = 120
# height_r = 90
# 设置裁剪后的resize长宽高
width_r = 70
height_r = 62
# 设置阈值
thresh = 100
# 测试集存放的位置
saveFile_img = "test_img/test_"
saveFile_vec = "test_vec/test_"
# 高斯函数参数
c = 2500.0
a = 15


# 开启摄像头 拍摄获得图片和向量
def get_test_img():
    cap = cv2.VideoCapture(0)  # 打开摄像机
    flag = 1  # 播放视频
    count = 0  # 记录的次数
    while cap.isOpened():  # 当摄像头打开时
        ret, frame = cap.read()  # 读取画面
        cv2.imshow('img', frame)  # 显示画面
        c = str(count)  # 转换成str型
        if cv2.waitKey(flag) == ord(' '):  # 按下空格键拍照
            resized_frame = cv2.resize(frame, (width_r, height_r))  # 调整图片大小
            new_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)  # RGB转灰度
            height, width = new_frame.shape[0:2]  # 得到frame的高度和宽度
            # 下面进行图片二值化
            for row in range(height):  # 遍历高度
                for col in range(width):  # 遍历宽度
                    gray = new_frame[row, col]  # 获取到灰度值
                    if gray > thresh:  # 大于阈值255
                        new_frame[row, col] = 255
                    else:  # 否则为0
                        new_frame[row, col] = 0
            cv2.imwrite(saveFile_img + c + '.jpg', new_frame)  # 保存测试的二值化图片
            image_arr = np.array(new_frame).reshape(width_r * height_r)  # 将其拼接为一维数组
            np.savetxt(saveFile_vec + c + ".txt", image_arr, delimiter=",", fmt='%d')  # 保存测试的数组
            count += 1  # 计数加一
            print("photo get")
            break  # 去掉此行可以拍多张照片
        if cv2.waitKey(flag) == ord('q'):  # 退出循环
            print("exit")
            break  # 退出循环
    cv2.destroyAllWindows()  # 关闭所有窗口
    cap.release()  # 释放摄像头


# Gaussian函数
def Gaussian(distance):
    weight = a * np.exp(-distance ** 2 / (2 * c ** 2))  # 权重计算方法
    return weight  # 返回权重


# 基与高斯函数改进的KNN算法
def Gauss_KNN(test_data, train_dataSet, labels, k):
    train_dataSetSize = train_dataSet.shape[0]  # 得到训练集的size
    distances_sq = ((np.subtract(np.tile(test_data, (train_dataSetSize, 1)), train_dataSet)) ** 2).sum(
        axis=1)  # 让测试数据与每一个训练数据相减 然后平方和相加
    distances = distances_sq ** 0.5  # 开方得到欧式距离
    sortedDist_index = distances.argsort()  # 距离排序的索引排序,并且返回下标顺序
    classCount = {}  # 生成空的类别计数字典
    for i in range(k):  # 遍历k个距离最近的值
        vote_label = labels[sortedDist_index[i]]  # 选择最小距离前k个的序号
        weight = Gaussian(distances[sortedDist_index[i]])  # 计算各自的高斯权重
        classCount[vote_label] = classCount.get(vote_label, 0) + 1 * weight  # 前k个每多1个则字典value+1，weigh为高斯权重
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 对字典value进行排序
    tenth = np.array(sortedDist_index)[:10]  # 加权距离最近的十个值
    list_pre = []  # 空数组，用于存放加权距离最近的十个字母
    for x in np.rint(tenth // 55 + 65):  # 序号用真除法除以55，然后加65转ascii码
        list_pre.append((chr(int(x))))  # int转chr，放入数组
    print("前十个候选为", list_pre)  # 输出数组
    return sortedClassCount[0][0] - 1  # 返回最有可能的情况


# 获得最好的k
def getBest_k():
    train_labels = []  # 训练集的标签矩阵
    train_FileList = listdir('label')  # 储存训练集label所有的文件名
    val_FileList = listdir('label_val')  # 储存验证集label_val所有的文件名
    m_train = len(train_FileList)  # 所有的训练数据的数量
    m_val = len(val_FileList)  # 所有的训练数据的数量
    train_Mat = np.zeros((m_train, width_r * height_r))  # 每行储存一个图像 0
    for i in range(m_train):  # 遍历所有训练集
        fileNameStr = train_FileList[i]  # 第i个文件的文件名
        fileStr = fileNameStr.split('.')[0]  # 分割“.”前后，得到类名
        classNumStr = int(fileStr.split('_')[0])  # 得到标签转为int
        train_labels.append(classNumStr)  # 训练集的标签矩阵
        train_Mat[i, :] = get_vector('label/%s' % fileNameStr)
    result_r = np.zeros(15)  # 建立大小为15值为0的数组，用于存放结果
    for k in range(1, 16):  # 遍历1-15个k值
        for i in range(m_val):  # 遍历所有验证集
            fileNameStr = val_FileList[i]  # 第i个文件的文件名
            test_Mat = get_vector("label_val/" + fileNameStr)  # 验证集的矩阵
            result = Gauss_KNN(test_Mat, train_Mat, train_labels, k)  # 调用Gauss_KNN函数进行验证集结果的计算
            print("predict result", result)  # 打印预测结果
            className = int(fileNameStr.split("_")[0]) - 1  # 得到真实的类名
            print("real result", className)  # 打印真实结果
            if result == className:  # 如果预测正确
                result_r[k - 1] += 1  # 则对应k的准确个数加一
            print(result_r)  # 打印该k值对应的准确结果数量
    print('准确率为', result_r / m_val)  # 打印该k值对应的准确率


# 该函数的主要功能是图像向量化，并返回向量
def get_vector(path):
    f = open(path, encoding="utf - 8")  # 读取数据
    row = f.readlines()  # 每一行的数据
    vector = []  # 数组初始化
    for line in row:  # 遍历每一行
        num = int(line.rstrip())  # str转int
        vector.append(num)  # 见数据保存到数组中
    f.close()  # 关闭文件
    return vector  # 返回向量数组


# 该函数的功能是进行预测
def predict():
    train_labels = []  # 训练集的标签矩阵
    train_FileList = listdir('label')  # 储存训练集label所有的文件名
    m = len(train_FileList)  # 所有的训练数据的数量
    train_Mat = np.zeros((m, width_r * height_r))  # 建立一个全为0的数组，每行储存一个图像
    for i in range(m):  # 遍历所有的训练文件
        fileNameStr = train_FileList[i]  # 第i个文件的文件名
        fileStr = fileNameStr.split('.')[0]  # 分割“.”前后，得到类名
        classNumStr = int(fileStr.split('_')[0])  # 得到标签转为int
        train_labels.append(classNumStr)  # 训练集的标签矩阵
        train_Mat[i, :] = get_vector('label/%s' % fileNameStr)  # 得到训练矩阵
    test_Mat = get_vector("test_vec/test_0.txt")  # 得到测试集
    test_myself_label = Gauss_KNN(test_Mat, train_Mat, train_labels, 4)  # 调用Gauss_KNN算法进行计算
    List_letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X', 'Y', 'Z']  # 把结果转化为对应的字母
    print("预测结果为 ", List_letter[test_myself_label])  # 打印输出结果


if __name__ == '__main__':
    # get_test_img()   #获取测试集
    # getBest_k()  # 获得最好的k
    predict()  # 预测
