from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

def Linear_transformation(filename):
    image_0 = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    plt.subplot(221)
    plt.imshow(image_0, vmin=0, vmax=255, cmap='gray')

    img_np = np.array(image_0)
    width, height = img_np.shape
    N = np.zeros(shape=(1, 256), dtype='float')  # 构造零矩阵，用以统计像素数

    # 遍历各个灰度值统计个数
    for i in range(0, width):
        for j in range(0, height):
            k = img_np[i, j]
            N[0][k] = N[0][k] + 1

    N = N.flatten()  # 扁平化

    plt.subplot(222)
    plt.bar([i for i in range(0, 256)], height=N, width=1)

    # 线性变化
    plt.subplot(223)
    J = img_np.astype('float')
    J = 0 + (J - 42) * (255 - 0) / (232 - 42)  # 利用公式转换
    for i in range(0, width):
        for j in range(0, height):
            if J[i, j] < 0:
                J[i, j] = 0
            elif J[i, j] > 255:
                J[i, j] = 255
    image_1 = J.astype('uint8')
    plt.imshow(image_1, vmin=0, vmax=255, cmap='gray')

    plt.subplot(224)
    J = J.flatten()
    plt.hist(J, bins=255)

    plt.show()

def Contrast_reversal(filename):
    img = cv2.imread(filename)

    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height = grayImage.shape[0]
    width = grayImage.shape[1]

    result = np.zeros((height, width), np.uint8)

    for i in range(height):
        for j in range(width):
            gray = 255 - grayImage[i, j]
            result[i, j] = np.uint8(gray)


    cv2.imshow("Gray Image", grayImage)
    cv2.imshow("Result", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Piecewise_linear_transformation(filename):
    image_0 = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    plt.subplot(321)
    plt.title('原图')
    plt.imshow(image_0, vmin=0, vmax=255, cmap='gray')

    img_np = np.array(image_0)
    width, height = img_np.shape
    N = np.zeros(shape=(1, 256), dtype='float')  # 构造零矩阵，用以统计像素数

    # 遍历各个灰度值统计个数
    for i in range(0, width):
        for j in range(0, height):
            k = img_np[i, j]
            N[0][k] = N[0][k] + 1

    N = N.flatten()  # 扁平化

    plt.subplot(322)
    plt.title('灰度分布直方图')
    plt.bar([i for i in range(0, 256)], height=N, width=1)

    # 灰度变换-线性变化
    plt.subplot(323)
    plt.title('线性变换')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    J = img_np.astype('float')
    J = 0 + (J - 42) * (255 - 0) / (232 - 42)  # 利用公式转换
    for i in range(0, width):
        for j in range(0, height):
            if J[i, j] < 0:
                J[i, j] = 0
            elif J[i, j] > 255:
                J[i, j] = 255
    image_1 = J.astype('uint8')
    plt.imshow(image_1, vmin=0, vmax=255, cmap='gray')

    plt.subplot(324)
    J = J.flatten()
    plt.hist(J, bins=255)

    # 灰度变换-分段线性变换
    Q = img_np.astype('float')
    for i in range(0, width):
        for j in range(0, height):
            if 0 <= Q[i, j] < 100:
                Q[i, j] = 0 + Q[i, j] * (50 / 100)
            elif 100 <= Q[i, j] < 200:
                Q[i, j] = 50 + (Q[i, j] - 100) * (200 - 50) / (200 - 100)
            elif 200 <= Q[i, j] < 255:
                Q[i, j] = 200 + (Q[i, j] - 200) * (255 - 200) / (255 - 200)

    for i in range(0, width):
        for j in range(0, height):
            if Q[i, j] <= 0:
                Q[i, j] = 0
            elif Q[i, j] >= 255:
                Q[i, j] = 255

    image_2 = Q.astype('uint8')
    plt.subplot(325)
    plt.title('分段线性变换')
    plt.imshow(image_2, vmin=0, vmax=255, cmap='gray')
    plt.subplot(326)
    plt.hist(Q.flatten(), bins=255)
    plt.show()


