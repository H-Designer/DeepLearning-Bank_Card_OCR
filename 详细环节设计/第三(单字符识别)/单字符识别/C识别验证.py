import os
import tensorflow as tf
from PIL import Image
from nets2 import nets_factory
import numpy as np
import matplotlib.pyplot as plt
# 不同字符数量
CHAR_SET_LEN = 10
# 图片高度
IMAGE_HEIGHT = 60
# 图片宽度
IMAGE_WIDTH = 160
# 批次
BATCH_SIZE = 1
# tfrecord文件存放路径
TFRECORD_FILE = r"C:\workspace\Python\Bank_Card_OCR\demo\test_result\tfrecords/1.tfrecords"

# placeholder
x = tf.placeholder(tf.float32, [None, 224, 224])

os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config = config)

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
image_batch, image_raw_batch, label_batch0 = tf.train.shuffle_batch(
    [image, image_raw, label0], batch_size=BATCH_SIZE,
    capacity=50000, min_after_dequeue=10000, num_threads=1)


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
    saver.restore(sess, '../Cmodels/model/crack_captcha1.model-6000')
    # saver.restore(sess, '../1/crack_captcha1.model-2500')

    # 创建一个协调器，管理线程
    coord = tf.train.Coordinator()
    # 启动QueueRunner, 此时文件名队列已经进队
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(6):
        # 获取一个批次的数据和标签
        b_image, b_image_raw, b_label0 = sess.run([image_batch,image_raw_batch,label_batch0])
        # 显示图片
        img = Image.fromarray(b_image_raw[0], 'L')
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        # 打印标签
        print('label:', b_label0)
        # 预测
        label0 = sess.run([predict0], feed_dict={x: b_image})
        # 打印预测值

        print('predict:', label0[0])
        # 通知其他线程关闭
    coord.request_stop()
    # 其他所有线程关闭之后，这一函数才能返回
    coord.join(threads)






#
# import tensorflow as tf
# import inference
#
# image_size = 224  # 输入层图片大小
#
# # 模型保存的路径和文件名
# MODEL_SAVE_PATH = "../Cmodels/"
# MODEL_NAME = "crack_captcha1.model-2500"
#
# # 加载需要预测的图片
# image_data = tf.gfile.FastGFile("01.png", 'rb').read()
#
# # 将图片格式转换成我们所需要的矩阵格式，第二个参数为1，代表1维
# decode_image = tf.image.decode_png(image_data, 1)
#
# # 再把数据格式转换成能运算的float32
# decode_image = tf.image.convert_image_dtype(decode_image, tf.float32)
#
# # 转换成指定的输入格式形状
# image = tf.reshape(decode_image, [-1, image_size, image_size, 1])
#
# # 定义预测结果为logit值最大的分类，这里是前向传播算法，也就是卷积层、池化层、全连接层那部分
# test_logit = inference.inference(image, train=False, regularizer=None)
#
# # 利用softmax来获取概率
# probabilities = tf.nn.softmax(test_logit)
#
# # 获取最大概率的标签位置
# correct_prediction = tf.argmax(test_logit, 1)
#
# # 定义Savar类
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
#
#     # 加载检查点状态，这里会获取最新训练好的模型
#     ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
#     if ckpt and ckpt.model_checkpoint_path:
#         # 加载模型和训练好的参数
#         saver.restore(sess, ckpt.model_checkpoint_path)
#         print("加载模型成功：" + ckpt.model_checkpoint_path)
#
#         # 通过文件名得到模型保存时迭代的轮数.格式：model.ckpt-6000.data-00000-of-00001
#         global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#
#         # 获取预测结果
#         probabilities, label = sess.run([probabilities, correct_prediction])
#
#         # 获取此标签的概率
#         probability = probabilities[0][label]
#
#         print("After %s training step(s),validation label = %d, has %g probability" % (global_step, label, probability))
#     else:
#         print("模型加载失败！" + ckpt.model_checkpoint_path)
#
#
#
# import os
# from tensorflow.python import pywrap_tensorflow
#
# current_path = os.getcwd()
# model_dir = os.path.join(current_path, '../Cmodels/')
# checkpoint_path = os.path.join(model_dir,'crack_captcha1.model-2500') # 保存的ckpt文件名，不一定是这个
# # Read data from checkpoint file
# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# # Print tensor name and values
# for key in var_to_shape_map:
#     print("tensor_name: ", key)
#
