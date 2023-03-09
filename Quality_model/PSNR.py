import cv2
import math
import numpy
import os

def psnr1(img1, img2):
    # compute mse
    # mse = np.mean((img1-img2)**2)
    mse = numpy.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    # compute psnr
    if mse < 1e-10:
        return 100
    psnr1 = 20 * math.log10(255 / math.sqrt(mse))
    return psnr1


def psnr2(img1, img2):
    mse = numpy.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1e-10:
        return 100
    psnr2 = 20 * math.log10(1 / math.sqrt(mse))
    return psnr2

