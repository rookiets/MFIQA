import numpy as np
from feature_extraction import color_moments, hog, GLCM, lbp
from scipy.spatial import distance
import cv2
import os


class AHP:
    """
    相关信息的传入和准备
    """

    def __init__(self, array):
        ## 记录矩阵相关信息
        self.array = array
        ## 记录矩阵大小
        self.n = array.shape[0]
        # 初始化RI值，用于一致性检验
        self.RI_list = [0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58,
                        1.59]
        # 矩阵的特征值和特征向量
        self.eig_val, self.eig_vector = np.linalg.eig(self.array)
        # 矩阵的最大特征值
        self.max_eig_val = np.max(self.eig_val)
        # 矩阵最大特征值对应的特征向量
        self.max_eig_vector = self.eig_vector[:, np.argmax(self.eig_val)].real
        # 矩阵的一致性指标CI
        self.CI_val = (self.max_eig_val - self.n) / (self.n - 1)
        # 矩阵的一致性比例CR
        self.CR_val = self.CI_val / (self.RI_list[self.n - 1])

    """
    一致性判断
    """

    def test_consist(self):
        # 打印矩阵的一致性指标CI和一致性比例CR
        print("判断矩阵的CI值为：" + str(self.CI_val))
        print("判断矩阵的CR值为：" + str(self.CR_val))
        # 进行一致性检验判断
        if self.n == 2:  # 当只有两个子因素的情况
            print("仅包含两个子因素，不存在一致性问题")
        else:
            if self.CR_val < 0.1:  # CR值小于0.1，可以通过一致性检验
                print("判断矩阵的CR值为" + str(self.CR_val) + ",通过一致性检验")
                return True
            else:  # CR值大于0.1, 一致性检验不通过
                print("判断矩阵的CR值为" + str(self.CR_val) + "未通过一致性检验")
                return False

    """
    算术平均法求权重
    """

    def cal_weight_by_arithmetic_method(self):
        # 求矩阵的每列的和
        col_sum = np.sum(self.array, axis=0)
        # 将判断矩阵按照列归一化
        array_normed = self.array / col_sum
        # 计算权重向量
        array_weight = np.sum(array_normed, axis=1) / self.n
        # 打印权重向量
        print("算术平均法计算得到的权重向量为：\n", array_weight)
        # 返回权重向量的值
        return array_weight

    """
    几何平均法求权重
    """

    def cal_weight__by_geometric_method(self):
        # 求矩阵的每列的积
        col_product = np.product(self.array, axis=0)
        # 将得到的积向量的每个分量进行开n次方
        array_power = np.power(col_product, 1 / self.n)
        # 将列向量归一化
        array_weight = array_power / np.sum(array_power)
        # 打印权重向量
        print("几何平均法计算得到的权重向量为：\n", array_weight)
        # 返回权重向量的值
        return array_weight

    """
    特征值法求权重
    """

    def cal_weight__by_eigenvalue_method(self):
        # 将矩阵最大特征值对应的特征向量进行归一化处理就得到了权重
        array_weight = self.max_eig_vector / np.sum(self.max_eig_vector)
        # 打印权重向量
        print("特征值法计算得到的权重向量为：\n", array_weight)
        # 返回权重向量的值
        return array_weight

    def put(self, path, pic1, pic2):
        total_path1 = path + pic1
        total_path2 = path + pic2
        # 计算颜色相似度
        result1 = color_moments.color_moments(total_path1)
        result2 = color_moments.color_moments(total_path2)
        fact1 = 1 / distance.euclidean(result1, result2)
        # 计算局部特征相似度
        gray = cv2.imread(total_path1, cv2.IMREAD_GRAYSCALE)
        org_lbp = lbp.origin_LBP(total_path2)
        gray = cv2.resize(gray, (384, 384))
        org_lbp = cv2.resize(org_lbp, (384, 384))
        np.set_printoptions(threshold=np.inf)
        fact2 = np.mean(gray.dot(org_lbp) / (np.linalg.norm(gray) * np.linalg.norm(org_lbp)))
        # 计算对比度和逆方差，纹理特征相似度
        result3 = GLCM.mytest(total_path2)
        fact3 = result3[2] * 0.5 + result3[4] * 0.5
        # 计算主体形状相似度
        img = cv2.imread(total_path1, cv2.IMREAD_GRAYSCALE)
        img_1 = cv2.imread(total_path2, cv2.IMREAD_GRAYSCALE)
        if (img is None):
            print('Not read image.')
        resizeimg = cv2.resize(img, (128, 64), interpolation=cv2.INTER_CUBIC)
        resizeimg_1 = cv2.resize(img_1, (128, 64), interpolation=cv2.INTER_CUBIC)
        cell_w = 8
        cell_x = int(resizeimg.shape[0] / cell_w)  # cell行数
        cell_y = int(resizeimg.shape[1] / cell_w)  # cell列数
        gammaimg = hog.gamma(resizeimg) * 255
        gammaimg_1 = hog.gamma(resizeimg_1) * 255
        feature1 = hog.hog(gammaimg, cell_x, cell_y, cell_w)
        feature2 = hog.hog(gammaimg_1, cell_x, cell_y, cell_w)
        fact4 = np.mean(feature1.dot(feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2)))
        # AHP判断矩阵
        b = np.array([[1, 1 / 2, 1 / 3, 1 / 4], [2, 1, 1 / 2, 1 / 3],
                      [3, 2, 1, 1 / 2], [4, 3, 2, 1]])
        # 算术平均法求权重
        weight1 = AHP(b).cal_weight_by_arithmetic_method()
        # 几何平均法求权重
        weight2 = AHP(b).cal_weight__by_geometric_method()
        # 特征值法求权重
        weight3 = AHP(b).cal_weight__by_eigenvalue_method()
        # 一致性检验
        AHP(b).test_consist()
        num1 = weight1[0]
        num2 = weight1[1]
        num3 = weight1[2]
        num4 = weight1[3]
        # 计算综合分数
        score = num1 * fact1 + num2 * fact2 + num3 * fact3 + num4 * fact4
        return score
        # print("图像综合分数为：", score)

    def Gamma_GRAY(picture_path):  # Gamma校正灰度化：Gray=
        # 读取图像
        img = cv2.imread(picture_path)
        # 获取图像尺寸
        h, w = img.shape[0:2]
        # 自定义空白单通道图像，用于存放灰度图
        gray = np.zeros((h, w), dtype=img.dtype)
        # 对原图像进行遍历，然后分别对B\G\R按比例灰度化
        for i in range(h):
            for j in range(w):
                a = img[i, j, 2] ** (2.2) + 1.5 * img[i, j, 1] ** (2.2) + 0.6 * img[i, j, 0] ** (2.2)  # 分子
                b = 1 + 1.5 ** (2.2) + 0.6 ** (2.2)  # 分母
                gray[i, j] = pow(a / b, 1.0 / 2.2)  # 开2.2次方根
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)  # BGR转换为RGB显示格式，方便通过matplotlib进行图像显示
        return gray


array1 = np.array([[1, 1 / 3, 1 / 5, 1 / 8], [3, 1, 1 / 2, 1 / 5],
                   [5, 2, 1, 1 / 3], [8, 5, 3, 1]])
a = AHP(array1)



