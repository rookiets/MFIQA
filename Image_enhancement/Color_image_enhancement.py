from PIL import Image
from PIL import ImageEnhance
import matplotlib.pyplot as plt

# 亮度增强
def enhance_brightness(image):
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened
#色度增强
def enhance_color(image):
    enh_col = ImageEnhance.Color(image)
    color = 1.5
    image_colored = enh_col.enhance(color)
    return image_colored
#对比度增强
def enhance_contrast(image):
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted
#锐度增强
def enhance_sharpness(image):
    enh_sha = ImageEnhance.Sharpness(image)
    sharpness = 3.0
    image_sharped = enh_sha.enhance(sharpness)
    return image_sharped

