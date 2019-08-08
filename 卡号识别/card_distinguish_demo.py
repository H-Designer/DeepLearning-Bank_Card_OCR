import os
import tensorflow as tf
from PIL import Image
from lib.nets2 import nets_factory
import numpy as np
import matplotlib.pyplot as plt
import cv2

## 读取图像，解决imread不能读取中文路径的问题
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    ##cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img
def distinguish(filename,test_images_dir):

    # 不同字符数量
    CHAR_SET_LEN = 10
    # 批次
    dir = 'test_result/cut/'+filename.split('.')[0]
    BATCH_SIZE = s=len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
    # tfrecord文件存放路径
    TFRECORD_FILE = "test_result/tfrecords/"+filename.split('.')[0]+'.tfrecords'
    # placeholder
    x = tf.placeholder(tf.float32, [None, 224, 224])

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定第一块GPU可用
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
    config.gpu_options.allow_growth = True  # 程序按需申请内存
    sess = tf.Session(config=config)

    # 从tfrecord读出数据
    def read_and_decode(filename):
        # 根据文件名生成一个队列
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TFRecordReader()
        # 返回文件名和文件
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'image': tf.FixedLenFeature([], tf.string),
                                               'label0': tf.FixedLenFeature([], tf.int64),

                                           })
        # 获取图片数据
        image = tf.decode_raw(features['image'], tf.uint8)
        # 没有经过预处理的灰度图
        image_raw = tf.reshape(image, [224, 224])
        # tf.train.shuffle_batch必须确定shape
        image = tf.reshape(image, [224, 224])
        # 图片预处理
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        # 获取label
        label0 = tf.cast(features['label0'], tf.int32)

        return image, image_raw, label0

    # 获取图片数据和标签
    image, image_raw, label0 = read_and_decode(TFRECORD_FILE)
    # 使用shuffle_batch可以随机打乱
    image_batch, image_raw_batch, label_batch0 = tf.train.batch(
        [image, image_raw, label0], batch_size=BATCH_SIZE,
        capacity=50000,  num_threads=1)
    # 定义网络结构
    train_network_fn = nets_factory.get_network_fn(
        'alexnet_v2',
        num_classes=CHAR_SET_LEN * 1,
        weight_decay=0.0005,
        is_training=False)

    with tf.Session() as sess:
        # inputs: a tensor of size [batch_size, height, width, channels]
        X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])
        # 数据输入网络得到输出值
        logits, end_points = train_network_fn(X)
        # 预测值
        logits0 = tf.slice(logits, [0, 0], [-1, 10])

        predict0 = tf.argmax(logits0, 1)

        # 初始化
        sess.run(tf.global_variables_initializer())
        # 载入训练好的模型
        saver = tf.train.Saver()
        saver.restore(sess, 'lib/models/model_distinguish/crack_captcha1.model-6000')
        # saver.restore(sess, '../1/crack_captcha1.model-2500')

        # 创建一个协调器，管理线程
        coord = tf.train.Coordinator()
        # 启动QueueRunner, 此时文件名队列已经进队
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(1):
            # 获取一个批次的数据和标签
            b_image, b_image_raw, b_label0 = sess.run([image_batch, image_raw_batch, label_batch0])
            # print(b_image_raw)
            # 显示图片
            img = Image.fromarray(b_image_raw[0], 'L')
            # plt.imshow(img)
            plt.axis('off')
            # plt.show()
            # 打印标签
            # print('lable',b_label0)
            distance = (b_label0[1]-b_label0[0])*1.4
            lb = []
            for i in range(len(b_label0)-1):
                if distance<=b_label0[i+1]-b_label0[i]:
                    lb.append(i)
            # print('lb',lb)
            # 预测
            label0 = sess.run([predict0], feed_dict={x: b_image})
            # 打印预测值
            predict = str(label0[0]).strip('[[]]').replace(' ','')
            predict1 = list(predict)  # str -> list
            # print(predict1)
            # print(lb)
            for i in range(len(lb)):
                yuan = predict1[lb[i]]
                predict1[lb[i]] = yuan+'_'
                # predict1.insert(lb[i], '_')  # 注意不用重新赋值
            predict = ''.join(predict1)  # list -> str
            print('predict:', predict)

            # 给train数据集打标签
            # 通知其他线程关闭
        coord.request_stop()
        # 其他所有线程关闭之后，这一函数才能返回
        coord.join(threads)
if __name__ == '__main__':
    test_images_dir = ''#此处填写定位图片的文件夹
    for filename in os.listdir(test_images_dir):
        distinguish(filename,test_images_dir)
        tf.reset_default_graph()#重置tensorflow的旧
