import os
from PIL import Image
from PIL import ImageEnhance

"""
1、对比度：白色画面(最亮时)下的亮度除以黑色画面(最暗时)下的亮度；
2、色彩饱和度：：彩度除以明度，指色彩的鲜艳程度，也称色彩的纯度；
3、色调：向负方向调节会显现红色，正方向调节则增加黄色。适合对肤色对象进行微调；
4、锐度：是反映图像平面清晰度和图像边缘锐利程度的一个指标。
"""
# 灰度处理


import random
def salt_and_pepper_noise(img, proportion=0.01):
    noise_img = img
    height, width = noise_img.shape[0], noise_img.shape[1]
    num = int(height * width * proportion)  # 多少个像素点添加椒盐噪声
    for i in range(num):
        w = random.randint(0, width - 1)
        h = random.randint(0, height - 1)
        if random.randint(0, 1) == 0:
            noise_img[h, w] = 0
        else:
            noise_img[h, w] = 255
    return noise_img

def makeL(dir):
    for filename in os.listdir(dir):
        image = Image.open(dir + filename)
        # 灰度图
        lim = image.convert('L')
        lim.save(dir + filename.split('.')[0] + 'L.png')

def augument(image_path, parent):
    # 读取图片
    image = Image.open(image_path)

    image_name = os.path.split(image_path)[1]
    name = os.path.splitext(image_name)[0]

    # 变亮

    # 亮度增强,增强因子为0.0将产生黑色图像；为1.0将保持原始图像。
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.6
    image_brightened1 = enh_bri.enhance(brightness)
    image_brightened1.save(os.path.join(parent, '{}_bri2.png'.format(name)))
    # 亮度增强,增强因子为0.0将产生黑色图像；为1.0将保持原始图像。
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.4
    image_brightened1 = enh_bri.enhance(brightness)
    image_brightened1.save(os.path.join(parent, '{}_bri3.png'.format(name)))

    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.2
    image_brightened1 = enh_bri.enhance(brightness)
    image_brightened1.save(os.path.join(parent, '{}_bri5.png'.format(name)))

    # 变暗
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 0.8
    image_brightened2 = enh_bri.enhance(brightness)
    image_brightened2.save(os.path.join(parent, '{}_bri8.png'.format(name)))

    # 变暗
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 0.6
    image_brightened2 = enh_bri.enhance(brightness)
    image_brightened2.save(os.path.join(parent, '{}_bri10.png'.format(name)))

    # 色度,增强因子为1.0是原始图像
    # 色度增强
    enh_col = ImageEnhance.Color(image)
    color = 1.4
    image_colored1 = enh_col.enhance(color)
    image_colored1.save(os.path.join(parent, '{}_col1.png'.format(name)))


    # 色度增强
    enh_col = ImageEnhance.Color(image)
    color = 1.2
    image_colored1 = enh_col.enhance(color)
    image_colored1.save(os.path.join(parent, '{}_col4.png'.format(name)))

    # 色度减弱
    enh_col = ImageEnhance.Color(image)
    color = 0.8
    image_colored1 = enh_col.enhance(color)
    image_colored1.save(os.path.join(parent, '{}_col7.png'.format(name)))
    # 色度减弱

    # 色度减弱
    enh_col = ImageEnhance.Color(image)
    color = 0.6
    image_colored1 = enh_col.enhance(color)
    image_colored1.save(os.path.join(parent, '{}_col9.png'.format(name)))

    # 对比度，增强因子为1.0是原始图片

    # 对比度增强
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.4
    image_contrasted1 = enh_con.enhance(contrast)
    image_contrasted1.save(os.path.join(parent, '{}_con2.png'.format(name)))

    # 对比度增强
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.2
    image_contrasted1 = enh_con.enhance(contrast)
    image_contrasted1.save(os.path.join(parent, '{}_con4.png'.format(name)))

    # 对比度减弱
    enh_con = ImageEnhance.Contrast(image)
    contrast = 0.8
    image_contrasted2 = enh_con.enhance(contrast)
    image_contrasted2.save(os.path.join(parent, '{}_con7.png'.format(name)))

    # 对比度减弱
    enh_con = ImageEnhance.Contrast(image)
    contrast = 0.6
    image_contrasted2 = enh_con.enhance(contrast)
    image_contrasted2.save(os.path.join(parent, '{}_con9.png'.format(name)))



    # 锐度，增强因子为1.0是原始图片
    # 锐度增强
    enh_sha = ImageEnhance.Sharpness(image)

    # 锐度增强

    sharpness = 3.0
    image_sharped1 = enh_sha.enhance(sharpness)
    image_sharped1.save(os.path.join(parent, '{}_sha7.png'.format(name)))
    enh_sha = ImageEnhance.Sharpness(image)
    # 锐度增强

    sharpness = 2.0
    image_sharped1 = enh_sha.enhance(sharpness)
    image_sharped1.save(os.path.join(parent, '{}_sha6.png'.format(name)))
    # 锐度增强

    sharpness = 1.5
    image_sharped1 = enh_sha.enhance(sharpness)
    image_sharped1.save(os.path.join(parent, '{}_sha5.png'.format(name)))
    # 锐度增强

    sharpness = 2.5
    image_sharped1 = enh_sha.enhance(sharpness)
    image_sharped1.save(os.path.join(parent, '{}_sha4.png'.format(name)))


    # 锐度减弱
    enh_sha = ImageEnhance.Sharpness(image)
    sharpness = 0.8
    image_sharped2 = enh_sha.enhance(sharpness)
    image_sharped2.save(os.path.join(parent, '{}_sha2.png'.format(name)))
    # 锐度减弱
    enh_sha = ImageEnhance.Sharpness(image)
    sharpness = 0.6
    image_sharped2 = enh_sha.enhance(sharpness)
    image_sharped2.save(os.path.join(parent, '{}_sha1.png'.format(name)))


dir = r'D:\比赛\软件杯(19年)\银行卡号识别\单位数据集\images\0/'

for parent, dirnames, filenames in os.walk(dir):
    for filename in filenames:
        fullpath = os.path.join(parent + '/', filename)
        if 'png'or'PNG' in fullpath:
            augument(fullpath, parent)
#             makeL(dir)
# for class_name in os.listdir(dir):
#     img = io.imread(dir+class_name)
#     noise_img = salt_and_pepper_noise(img)
#     io.imsave(dir + class_name.split('.')[0] + 'noise1.png', noise_img)
