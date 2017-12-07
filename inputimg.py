import tensorflow as tf
import random

from data_selection import *


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # 变量的初始值为截断正太分布
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """
    tf.nn.conv2d功能：给定4维的input和filter，计算出一个2维的卷积结果
    前几个参数分别是input, filter, strides, padding, use_cudnn_on_gpu, ...
    input   的格式要求为一个张量，[batch, in_height, in_width, in_channels],批次数，图像高度，图像宽度，通道数
    filter  的格式为[filter_height, filter_width, in_channels, out_channels]，滤波器高度，宽度，输入通道数，输出通道数
    strides 一个长为4的list. 表示每次卷积以后在input中滑动的距离
    padding 有SAME和VALID两种选项，表示是否要保留不完全卷积的部分。如果是SAME，则保留
    use_cudnn_on_gpu 是否使用cudnn加速。默认是True
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """
    tf.nn.max_pool 进行最大值池化操作,而avg_pool 则进行平均值池化操作
    几个参数分别是：value, ksize, strides, padding,
    value:  一个4D张量，格式为[batch, height, width, channels]，与conv2d中input格式一样
    ksize:  长为4的list,表示池化窗口的尺寸
    strides: 窗口的滑动值，与conv2d中的一样
    padding: 与conv2d中用法一样。
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 28,28])
x_image = tf.reshape(x, [-1,28,28,1])

"""
# 第一层
# 卷积核(filter)的尺寸是5*5, 通道数为1，输出通道为32，即feature map 数目为32
# 又因为strides=[1,1,1,1] 所以单个通道的输出尺寸应该跟输入图像一样。即总的卷积输出应该为?*28*28*32
# 也就是单个通道输出为28*28，共有32个通道,共有?个批次
# 在池化阶段，ksize=[1,2,2,1] 那么卷积结果经过池化以后的结果，其尺寸应该是？*14*14*32
"""
W_conv1 = weight_variable([5, 5, 1, 32])  # 卷积是在每个5*5的patch中算出32个特征，分别是patch大小，输入通道数目，输出通道数目
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.elu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

"""
# 第二层
# 卷积核5*5，输入通道为32，输出通道为64。
# 卷积前图像的尺寸为 ?*14*14*32， 卷积后为?*14*14*64
# 池化后，输出的图像尺寸为?*7*7*64
"""
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.elu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 第三层 是个全连接层,输入维数7*7*64, 输出维数为256
W_fc1 = weight_variable([7 * 7 * 64, 256])
b_fc1 = bias_variable([256])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.elu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# keep_prob = tf.placeholder(tf.float32) # 这里使用了drop out,即随机安排一些cell输出值为0，可以防止过拟合
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第四层，输入256维，输出512维
W_fc2 = weight_variable([256, 512])
b_fc2 = bias_variable([512])
h_fc1_flat = tf.reshape(h_fc1, [-1, 256])
h_fc2 = tf.nn.elu(tf.matmul(h_fc1_flat, W_fc2) + b_fc2)
keep_prob = tf.placeholder(tf.float32) # 这里使用了drop out,即随机安排一些cell输出值为0，可以防止过拟合
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

'''
setup: filename
'''
filename='facedata_pkl/IMM_face_data_40_5.pkl'
class_num=int(filename.split('_')[4])
cut_num=int(filename.split('_')[5].split('.')[0])
img_data = pkl.load(open(filename, 'rb'))
''''''

# 第五层，输入512维，输出?维 ：具体参考样本类别总数
W_fc3 = weight_variable([512, class_num])
b_fc3 = bias_variable([class_num])
y_conv=tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3) # 使用softmax作为多分类激活函数
y_ = tf.placeholder(tf.float32, [None, class_num])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])) # 损失函数，交叉熵
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # 使用adam优化
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) # 计算准确度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer()) # 变量初始化

train_acc_list=list()
iteration=300
test_acc_list=list()

for k in range(0,10):
    sess.run(tf.global_variables_initializer())  # 变量初始化
    feature=img_data[0]
    label=img_data[1]
    rand_data=list(zip(feature,label))
    random.shuffle(rand_data)
    feature[:],label[:]=zip(*rand_data)
    img_data_shuffled=tuple((feature,label))
    training_data,test_data=select_data(img_data_shuffled,class_num,cut_num)
    for i in range(1, iteration):
        for j in range(0, len(training_data[0]), 10):
            if j + 10 <= len(training_data[0]):
                data = training_data[0][j:j + 10]
                label = training_data[1][j:j + 10]
            else:
                data = training_data[0][len(training_data[0]) - 10:len(training_data[0])]
                label = training_data[1][len(training_data[0]) - 10:len(training_data[0])]
            data = np.array(data)
            label = np.array(label, float)
            train_step.run(feed_dict={x: data, y_: label, keep_prob: 0.5})

    test_accuracy = accuracy.eval(feed_dict={
        x: test_data[0], y_: test_data[1], keep_prob: 1.0})
    print("{}th times test acc: {}".format(k+1,test_accuracy*100))
    test_acc_list.append(test_accuracy*100)

print("10 times test acc:{} ,mean acc:{}".format(test_acc_list,np.mean(test_acc_list)))
