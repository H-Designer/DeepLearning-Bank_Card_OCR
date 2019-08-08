

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.lib.nets.vgg16 import vgg8
from lib.utils.timer import Timer

CLASSES = ('__background__','lb')
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}



## 读取图像，解决imread不能读取中文路径的问题
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    ##cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img
def position(test_images_dir,filename):
    print("正在进行定位处理")
    # 第一步
    def vis_detections(im, class_name, dets, thresh=0.5):
        """Draw detected bounding boxes."""
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            return
        im = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        sco = []
        for i in inds:
            score = dets[i, -1]
            sco.append(score)
        maxscore = max(sco)  # 成绩最大值
        for i in inds:
            score = dets[i, -1]
            if score == maxscore:
                bbox = dets[i, :4]
                img = cv_imread(test_images_dir + filename)
                sp = img.shape
                width = sp[1]
                if bbox[0] > 50 and bbox[2] + 50 < width:
                    cropped = img[int(bbox[1]):int(bbox[3]), int(bbox[0] - 50):int(bbox[2]) + 50]  # 裁剪坐标为[y0:y1, x0:x1]
                if bbox[0] < 50 and bbox[2] + 50 < width:
                    cropped = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]) + 50]  # 裁剪坐标为[y0:y1, x0:x1]
                if bbox[0] > 50 and bbox[2] + 50 > width:
                    cropped = img[int(bbox[1]):int(bbox[3]), int(bbox[0] - 50):int(bbox[2])]  # 裁剪坐标为[y0:y1, x0:x1]
                if bbox[0] < 50 and bbox[2] + 50 > width:
                    cropped = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]  # 裁剪坐标为[y0:y1, x0:x1]
                path = 'test_result/position/'#存定位图片地址
                if not os.path.exists(path):
                    os.makedirs(path)
                # 重定义图片的大小
                cv2.imwrite(path + filename, cropped)
                # 定位框体显示
                if bbox[0] > 50 and bbox[2] + 50 < width:
                    ax.add_patch(plt.Rectangle((bbox[0]-50, bbox[1]),
                                               bbox[2] - bbox[0]+100,
                                               bbox[3] - bbox[1], fill=False,
                                               edgecolor='red', linewidth=3.5))
                if bbox[0] < 50 and bbox[2] + 50 < width:
                    ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),
                                               bbox[2] - bbox[0]+50,
                                               bbox[3] - bbox[1], fill=False,
                                               edgecolor='red', linewidth=3.5))
                if bbox[0] > 50 and bbox[2] + 50 > width:
                    ax.add_patch(plt.Rectangle((bbox[0]-50, bbox[1]),
                                               bbox[2] - bbox[0]+50,
                                               bbox[3] - bbox[1], fill=False,
                                               edgecolor='red', linewidth=3.5))
                if bbox[0] < 50 and bbox[2] + 50 > width:
                    ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),
                                               bbox[2] - bbox[0],
                                               bbox[3] - bbox[1], fill=False,
                                               edgecolor='red', linewidth=3.5))
                #     每个框的成绩输出
                # ax.text(bbox[0], bbox[1] - 2,
                #         '{:s} {:.3f}'.format(class_name, score),
                #         bbox=dict(facecolor='blue', alpha=0.5),
                #         fontsize=14, color='white')
                ax.set_title(('{} detections with '
                              'p({} | box) >= {:.1f}').format(class_name, class_name, thresh),
                             fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.draw()

    def demo_posiotion(sess, net, image_name,path):
        """Detect object classes in an image using pre-computed object proposals."""

        # Load the demo image
        im_file = os.path.join(cfg.FLAGS2["data_dir"], path, image_name)
        im = cv_imread(im_file)
        print(im_file)
        scores, boxes = im_detect(sess, net, im)

        # Visualize detections for each class
        CONF_THRESH = 0.1
        NMS_THRESH = 0.1
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            # print(cls_scores)#一个300个数的数组
            # np.newaxis增加维度  np.hstack将数组拼接在一起
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]

            vis_detections(im, cls, dets, thresh=CONF_THRESH)

    def parse_args():
        """Parse input arguments."""
        parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
        parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                            choices=NETS.keys(), default='vgg16')
        parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                            choices=DATASETS.keys(), default='pascal_voc')
        args = parser.parse_args()

        return args
    args = parse_args()
    demonet = args.demo_net
    tfmodel = r'lib/models/model_position/vgg16_faster_rcnn_iter_8000.ckpt'
    # 路径异常提醒
    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", 2,
                        tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Demo for '+test_images_dir+'{}'.format(filename))
    demo_posiotion(sess, net, filename,test_images_dir)
    plt.savefig("test_result/"+filename)
    plt.show()


if __name__ == '__main__':
    test_images_dir = ''#此处填写定位图片的文件夹
    for filename in os.listdir(test_images_dir):
        position(test_images_dir,filename)
        tf.reset_default_graph()#重置tensorflow的旧变量，重置图标s