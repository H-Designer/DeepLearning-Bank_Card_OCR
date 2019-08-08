  # ******    图片处理    *********
import cv2
import os
import numpy as np
from PIL import Image
DATASET_DIR = 'demo/'
for filename in os.listdir(DATASET_DIR):
    print (filename)
    filename = cv2.imread(DATASET_DIR+filename)
    img = cv2.resize(filename, (200, 30))  # 调整图片分辨率
    # 灰度化处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('123', gray)
    cv2.waitKey(0)


# from PIL import Image
#
#
# # 二值化处理
# def two_value():
#   for i in range(1, 5):
#       # 打开文件夹中的图片
#       image = Image.open('data/img/0_00h_0.png')
#       # 灰度图
#       lim = image.convert('L')
#       # 灰度阈值设为165，低于这个值的点全部填白色
#       # threshold = 200
#       # table = []
#       # for j in range(256):
#       #     if j < threshold:
#       #         table.append(0)
#       #     else:
#       #         table.append(1)
#       #
#       # bim = lim.point(table, '1')
#       lim.save('data/img/qqqqq.png')
#
#
# two_value()


# ******************移动文件*****************

# import cv2
# import os
# from shutil import copyfile
#
# import numpy as np
# from PIL import Image
# DATASET_DIR = 'D:\比赛\软件杯(19年)\sample\Train\imgs/'
# for filename in os.listdir(DATASET_DIR):
#     if '_rono' in filename:
#         copyfile(DATASET_DIR+filename,'data/images/'+filename)
#
#

# *******************   批量灰度处理     *************************
# import cv2
# import os
# from shutil import copyfile
#
# import numpy as np
# from PIL import Image
# DATASET_DIR = 'D:\data\images/'
# for filename in os.listdir(DATASET_DIR):
#     image = Image.open(DATASET_DIR+filename)
#      # 灰度图
#     lim = image.convert('L')
#     lim.save(DATASET_DIR+filename.split('.')[0]+'L.png')


# ***********   处理字符串
# import re
#
# files = '6000a0rono15bri5.png'
# lable = re.split(" |a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z", files)[0].replace('_','')
# print(lable)
# # import re
# #
# # a = "Hello world!How are you?My friend.Tom"
# #
# # print(re.split(" |!|\?|\.", a))




# ******************    chulitupian*******************
# import  cv2
# import numpy as np
# f = open("data/sample/Test/sample.txt", "r")
# while True:
#     line = f.readline()
#     if line:
#         pass    # do something here
#         line=line.strip()
#         p=line.rfind(' ')
#         filename=line[0:p]
#         print (filename)
#
#         filename = cv2.imread(filename)
#         train_images = [cv2.resize(tmp, (10, 12)) for tmp in filename]  # 改变图像大小，目的：使图片大小统一
#         # train_images = [bytes(list(np.reshape(tmp, [10 * 12 * 3]))) for tmp in filename]  # 将图片这种整形数据转为bytes形式
#     else:
#         break
# f.close()
#
