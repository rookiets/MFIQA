import cv2
import numpy as np
import PIL.Image as Image
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat

def origin_LBP(img):
    dst = np.zeros(img.shape, dtype=img.dtype)
    h, w = img.shape
    start_index = 1
    for i in range(start_index, h-1):
        for j in range(start_index, w-1):
            center = img[i][j]
            code = 0
#             顺时针，左上角开始的8个像素点与中心点比较，大于等于的为1，小于的为0，最后组成8位2进制
            code |= (img[i-1][j-1] >= center) << (np.uint8)(7)
            code |= (img[i-1][j  ] >= center) << (np.uint8)(6)
            code |= (img[i-1][j+1] >= center) << (np.uint8)(5)
            code |= (img[i  ][j+1] >= center) << (np.uint8)(4)
            code |= (img[i+1][j+1] >= center) << (np.uint8)(3)
            code |= (img[i+1][j  ] >= center) << (np.uint8)(2)
            code |= (img[i+1][j-1] >= center) << (np.uint8)(1)
            code |= (img[i  ][j-1] >= center) << (np.uint8)(0)
            dst[i-start_index][j-start_index]= code
    return dst
# 将图片转换为矩阵
def image_to_matrix(file_name):
    # 读取图片
    image = Image.open(file_name)
    # 显示图片
    #image.show()
    width, height = image.size
    # 灰度化
    image_grey = image.convert("L")
    data = image_grey.getdata()
    data = np.matrix(data, dtype="float") / 255.0
    new_data = np.reshape(data, (height, width))
    return new_data

def compareHist( stdimg, ocimg):
    stdimg = cv2.imread(str(stdimg), 0)
    ocimg = cv2.imread(str(ocimg), 0)
    stdimg = np.float32(stdimg)
    ocimg = np.float32(ocimg)
    stdimg = np.ndarray.flatten(stdimg)
    ocimg = np.ndarray.flatten(ocimg)
    imgocr = np.corrcoef(stdimg, ocimg)
    print(imgocr[0, 1])
    return imgocr[0, 1] > 0.96



