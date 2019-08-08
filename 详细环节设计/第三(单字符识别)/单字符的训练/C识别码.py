import os
import tensorflow as tf
from PIL import Image
from lib.nets2 import nets_factory
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config = config)



# 不同字符数量
CHAR_SET_LEN = 10
# 图片高度
IMAGE_HEIGHT = 60
# 图片宽度
IMAGE_WIDTH = 160
# 批次
BATCH_SIZE = 28
# tfrecord文件存放路径
TFRECORD_FILE = "C:\workspace\Python\deep-learning\card\Cimages/train.tfrecords"

# placeholder
x = tf.placeholder(tf.float32, [None, 224, 224])
y0 = tf.placeholder(tf.float32, [None])

# 学习率
lr = tf.Variable(0.001, dtype=tf.float32)


# 从tfrecord读出数据
def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image' : tf.FixedLenFeature([], tf.string),
                                           'label0': tf.FixedLenFeature([], tf.int64),
                                       })
    # 获取图片数据
    image = tf.decode_raw(features['image'], tf.uint8)
    # tf.train.shuffle_batch必须确定shape
    image = tf.reshape(image, [224, 224])
    # 图片预处理
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    # 获取label
    label0 = tf.cast(features['label0'], tf.int32)
    return image, label0


# 获取图片数据和标签
image, label0 = read_and_decode(TFRECORD_FILE)

# 使用shuffle_batch可以随机打乱
image_batch, label_batch0= tf.train.shuffle_batch(
    [image, label0], batch_size=BATCH_SIZE,
    capacity=50000, min_after_dequeue=10000, num_threads=1)

# 定义网络结构
train_network_fn = nets_factory.get_network_fn(
    'alexnet_v2',
    num_classes=CHAR_SET_LEN * 1,
    weight_decay=0.0005,
    is_training=True)

with tf.Session() as sess:
    # inputs: a tensor of size [batch_size, height, width, channels]
    X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])
    # 数据输入网络得到输出值
    logits, end_points = train_network_fn(X)

    # 把标签转成one_hot的形式
    one_hot_labels0 = tf.one_hot(indices=tf.cast(y0, tf.int32), depth=CHAR_SET_LEN)

    # 把标签转成长度为40的向量
    label_40 = tf.concat([one_hot_labels0], 1)
    # 计算loss
    loss_40 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label_40))
    # 优化loss
    optimizer_40 = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_40)
    # 计算准确率
    correct_prediction_40 = tf.equal(tf.argmax(label_40, 1), tf.argmax(logits, 1))
    accuracy_40 = tf.reduce_mean(tf.cast(correct_prediction_40, tf.float32))

    # 用于保存模型
    saver = tf.train.Saver()
    # 初始化
    sess.run(tf.global_variables_initializer())

    # 创建一个协调器，管理线程
    coord = tf.train.Coordinator()
    # 启动QueueRunner, 此时文件名队列已经进队
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(3500):
        # 获取一个批次的数据和标签
        b_image, b_label0 = sess.run([image_batch, label_batch0])
        # 优化模型
        sess.run(optimizer_40, feed_dict={x: b_image, y0: b_label0})
        # 每迭代20次计算一次loss和准确率
        if i % 4 == 0:
            # 每迭代3000次降低一次学习率
            if i % 400 == 0:
                sess.run(tf.assign(lr, lr / 3))
            acc, loss_ = sess.run([accuracy_40, loss_40], feed_dict={x: b_image,
                                                                     y0: b_label0})
            learning_rate = sess.run(lr)
            print("Iter:%d  Loss:%.4f  Accuracy:%.4f  Learning_rate:%.4f" % (i, loss_, acc, learning_rate))

            #             acc0,acc1,acc2,acc3,loss_ = sess.run([accuracy0,accuracy1,accuracy2,accuracy3,total_loss],feed_dict={x: b_image,
            #                                                                                                                 y0: b_label0,
            #                                                                                                                 y1: b_label1,
            #                                                                                                                 y2: b_label2,
            #                                                                                                                 y3: b_label3})
            #             learning_rate = sess.run(lr)
            #             print ("Iter:%d  Loss:%.3f  Accuracy:%.2f,%.2f,%.2f,%.2f  Learning_rate:%.4f" % (i,loss_,acc0,acc1,acc2,acc3,learning_rate))

            # 保存模型
            if i == 3300:
                saver.save(sess, "C:\workspace\Python\deep-learning\card\Cmodels/crack_captcha1.model", global_step=i)
                break
                # 通知其他线程关闭
    coord.request_stop()
    # 其他所有线程关闭之后，这一函数才能返回
    coord.join(threads)