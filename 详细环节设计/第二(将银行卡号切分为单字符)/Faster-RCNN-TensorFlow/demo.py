#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer

os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config = config)

CLASSES = ('__background__','lb')
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    sco=[]
    for i in inds:
        score = dets[i, -1]
        sco.append(score)
    maxscore=max(sco)
    # print(maxscore)成绩最大值
    for i in inds:
        # print(i)
        score = dets[i, -1]
        if score==maxscore:
            bbox = dets[i, :4]
            # print(bbox)#目标框的4个坐标
            img = cv2.imread("data/demo/"+filename)
            # img = cv2.imread('data/demo/000002.jpg')
            sp=img.shape
            width = sp[1]
            if bbox[0]>20 and bbox[2]+20<width:
                cropped = img[int(bbox[1]):int(bbox[3]), int(bbox[0]-20):int(bbox[2])+20] # 裁剪坐标为[y0:y1, x0:x1]
            if bbox[0]<20 and bbox[2]+20<width:
                cropped = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])+20] # 裁剪坐标为[y0:y1, x0:x1]
            if bbox[0] > 20 and bbox[2] + 20 > width:
                cropped = img[int(bbox[1]):int(bbox[3]), int(bbox[0] - 20):int(bbox[2])]  # 裁剪坐标为[y0:y1, x0:x1]
            path = 'cut1/'
            # 重定义图片的大小
            res = cv2.resize(cropped, (1000, 100), interpolation=cv2.INTER_CUBIC)  # dsize=（2*width,2*height）
            cv2.imwrite(path+str(i)+filename, res)
            ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

            ax.set_title(('{} detections with '
                          'p({} | box) >= {:.1f}').format(class_name, class_name,thresh),
                         fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo', image_name)
    im = cv2.imread(im_file)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.1
    NMS_THRESH = 0.1
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        # print(cls_scores)#一个300个数的数组
        #np.newaxis增加维度  np.hstack将数组拼接在一起
        dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
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


if __name__ == '__main__':
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset

    #tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])
    tfmodel = r'./default/voc_2007_trainval/cut1/vgg16_faster_rcnn_iter_8000.ckpt'
    # 路径异常提醒
    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    # elif demonet == 'res101':
        # net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", 2,
                        tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))
    # # 文件夹下所有图片进行识别
    # for filename in os.listdir(r'data/demo/'):
    #     im_names = [filename]
    #     for im_name in im_names:
    #         print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #         print('Demo for data/demo/{}'.format(im_name))
    #         demo(sess, net, im_name)
    #
    #     plt.show()
    # 单一图片进行识别
    filename = '0001.jpg'
    im_names = [filename]
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name)
    plt.show()
