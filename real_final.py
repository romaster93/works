import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from keras import backend
from tensorflow.keras.datasets.cifar10 import load_data

(x_train, y_train), (x_test, y_test) = load_data()

x_train_2 = x_train.astype('float32') / 255.0
x_test_2 = x_test.astype('float32') / 255.0


def RGB2Gray(img, fmt):
    if fmt == 'channels_first':
        R = img[:, 0:1]
        G = img[:, 1:2]
        B = img[:, 2:3]

    else:
        R = img[..., 0:1]
        G = img[..., 1:2]
        B = img[..., 2:3]

    return 0.299 * R + 0.587 * G + 0.114 * B


x_train_gray = RGB2Gray(x_train_2, backend.image_data_format())
x_test_gray = RGB2Gray(x_test_2, backend.image_data_format())

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1], name='x')
y = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='y')
n_batch = tf.placeholder(tf.int32)
training = tf.placeholder_with_default(False, shape=[])

##첫번째 레이어 쌓기
W_conv1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 64], stddev=5e-2))  # 32
b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)  # 32
W_conv2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=5e-2))  # 32
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)  # 32
bn_conv2 = tf.layers.batch_normalization(h_conv2, momentum=0.9, training=True)

##MAX_POOLING
h_pool1 = tf.nn.max_pool(bn_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # 16

##두번째 레이어 쌓기
W_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2))  # 16
b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)  # 16
W_conv4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))  # 16
b_conv4 = tf.Variable(tf.constant(0.1, shape=[128]))
h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)  # 16
bn_conv4 = tf.layers.batch_normalization(h_conv4, momentum=0.9, training=True)
##MAX_POOLING
h_pool2 = tf.nn.max_pool(bn_conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # 8

##세번째 레이어 쌓기
W_conv5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], stddev=5e-2))  # 8
b_conv5 = tf.Variable(tf.constant(0.1, shape=[256]))
h_conv5 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5)  # 8
W_conv6 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], stddev=5e-2))  # 8
b_conv6 = tf.Variable(tf.constant(0.1, shape=[256]))
h_conv6 = tf.nn.relu(tf.nn.conv2d(h_conv5, W_conv6, strides=[1, 1, 1, 1], padding='SAME') + b_conv6)  # 8
bn_conv6 = tf.layers.batch_normalization(h_conv6, momentum=0.9,
                                         training=True)
##MAX_POOLING
h_pool3 = tf.nn.max_pool(bn_conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # 4

##네번째 레이어 쌓기
W_conv7 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 512], stddev=5e-2))  # 4
b_conv7 = tf.Variable(tf.constant(0.1, shape=[512]))
h_conv7 = tf.nn.relu(tf.nn.conv2d(h_pool3, W_conv7, strides=[1, 1, 1, 1], padding='SAME') + b_conv7)  # 4
W_conv8 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=5e-2))  # 4
b_conv8 = tf.Variable(tf.constant(0.1, shape=[512]))
h_conv8 = tf.nn.relu(tf.nn.conv2d(h_conv7, W_conv8, strides=[1, 1, 1, 1], padding='SAME') + b_conv8)  # 4
bn_conv8 = tf.layers.batch_normalization(h_conv8, momentum=0.9,
                                         training=True)

##MAX_POOLING
h_pool4 = tf.nn.max_pool(bn_conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # 2

##up conv 하기 직전
W_conv9 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 1024], stddev=5e-2))  # 2
b_conv9 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_conv9 = tf.nn.relu(tf.nn.conv2d(h_pool4, W_conv9, strides=[1, 1, 1, 1], padding='SAME') + b_conv9)  # 2
W_conv10 = tf.Variable(tf.truncated_normal(shape=[3, 3, 1024, 1024], stddev=5e-2))  # 2
b_conv10 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_conv10 = tf.nn.relu(tf.nn.conv2d(h_conv9, W_conv10, strides=[1, 1, 1, 1], padding='SAME') + b_conv10)  # 2
bn_conv10 = tf.layers.batch_normalization(h_conv10, momentum=0.9,
                                          training=True)

##upcov는 한칸만 해서 진행해보자
W_conv11 = tf.Variable(tf.truncated_normal(shape=[2, 2, 512, 1024], stddev=0.05))  # 2
b_conv11 = tf.Variable(tf.constant(0.1, shape=[512]))  ##쪼금 이상 아마도 맞을듯 근데 ㅇㅇ
h_conv11 = tf.nn.relu(tf.nn.conv2d_transpose(bn_conv10, W_conv11, output_shape=[n_batch, 4, 4, 512],
                                             strides=[1, 2, 2, 1]) + b_conv11)  # 4
bn_conv11 = tf.layers.batch_normalization(h_conv11, momentum=0.99
                                          , training=True)
# output_shape = [?,4,4,512]
##잘라붙이기
cat_11 = tf.concat([bn_conv8, bn_conv11], axis=-1)

##컨벌루션 진행
W_convp1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 1024, 512], stddev=5e-2))
b_convp1 = tf.Variable(tf.constant(0.1, shape=[512]))
h_convp1 = tf.nn.relu(tf.nn.conv2d(cat_11, W_convp1, strides=[1, 1, 1, 1], padding='SAME') + b_convp1)
bn_conv12 = tf.layers.batch_normalization(h_convp1, momentum=0.9,
                                          training=True)

##upcov두번째층
W_conv12 = tf.Variable(tf.truncated_normal(shape=[2, 2, 256, 512], stddev=0.05))
b_conv12 = tf.Variable(tf.constant(0.1, shape=[256]))
h_conv12 = tf.nn.relu(
    tf.nn.conv2d_transpose(bn_conv12, W_conv12, output_shape=[n_batch, 32 // 4, 32 // 4, 256], strides=[1, 2, 2, 1],
                           padding='SAME') + b_conv12)
bn_conv13 = tf.layers.batch_normalization(h_conv12
                                          , momentum=0.9, training=True)

##잘라붙이기
cat_12 = tf.concat([bn_conv6, bn_conv13], axis=-1)

##컨벌루션 진행
W_convp2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 256], stddev=5e-2))
b_convp2 = tf.Variable(tf.constant(0.1, shape=[256]))
h_convp2 = tf.nn.relu(tf.nn.conv2d(cat_12, W_convp2, strides=[1, 1, 1, 1], padding='SAME') + b_convp2)
bn_conv14 = tf.layers.batch_normalization(h_convp2, momentum=0.9,
                                          training=True)

##upcov세번째층
W_conv13 = tf.Variable(tf.truncated_normal(shape=[2, 2, 128, 256], stddev=0.05))
b_conv13 = tf.Variable(tf.constant(0.1, shape=[128]))
h_conv13 = tf.nn.relu(
    tf.nn.conv2d_transpose(bn_conv14, W_conv13, output_shape=[n_batch, 32 // 2, 32 // 2, 128], strides=[1, 2, 2, 1],
                           padding='SAME') + b_conv13)
bn_conv15 = tf.layers.batch_normalization(h_conv13, momentum=0.9
                                          , training=True)

##잘라붙이기
cat_13 = tf.concat([bn_conv4, bn_conv15], axis=-1)

##컨벌루션 진행
W_convp3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 128], stddev=5e-2))
b_convp3 = tf.Variable(tf.constant(0.1, shape=[128]))
h_convp3 = tf.nn.relu(tf.nn.conv2d(cat_13, W_convp3, strides=[1, 1, 1, 1], padding='SAME') + b_convp3)
bn_conv16 = tf.layers.batch_normalization(h_convp3, momentum=0.9,
                                          training=True)

##upcov네번째층
W_conv14 = tf.Variable(tf.truncated_normal(shape=[2, 2, 64, 128], stddev=5e-2))
b_conv14 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv14 = tf.nn.relu(
    tf.nn.conv2d_transpose(bn_conv16, W_conv14, output_shape=[n_batch, 32, 32, 64], strides=[1, 2, 2, 1],
                           padding='SAME') + b_conv14)
bn_conv17 = tf.layers.batch_normalization(h_conv14
                                          , momentum=0.9, training=True)

##잘라붙이기
cat_14 = tf.concat([bn_conv2, bn_conv17], axis=-1)

##컨벌루션 진행
W_convp4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 3], stddev=5e-2))
b_convp4 = tf.Variable(tf.constant(0.1, shape=[3]))
h_convp4 = tf.nn.conv2d(cat_14, W_convp4, strides=[1, 1, 1, 1], padding='SAME') + b_convp4  # 최종OUTPUT

##손실함수 활성화함수

loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=y, predictions=h_convp4))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    with tf.name_scope('train'):
        train_step = tf.train.AdadeltaOptimizer(0.0001).minimize(loss)

# train_step = tf.train.AdadeltaOptimizer(0.0001).minimize(loss)

# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops):
#   train_step = tf.train.AdadeltaOptimizer(0.0001).minimize(loss)


(x_train, y_train), (x_test, y_test) = load_data()

batch_size = 100

# 다 구성했으니까 시작한다
sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)

num_batch = int(len(x_train) / batch_size)
print("시작")
for epoch in range(500):
    for i in range(num_batch):
        start = i * batch_size
        end = start + batch_size
        _, cost, final_data = sess.run([train_step, loss, h_convp4],
                                       feed_dict={x: x_train_gray[start:end, :, :], y: x_train[start:end, :, :],
                                                  n_batch: batch_size,training: True})

        print(cost)

# saver.save(sess, 'my_test_model')


final_data = sess.run(h_convp4, feed_dict={x: x_test_gray[0].reshape(-1, 32, 32, 1), n_batch: 1})

plt.figure()
plt.imshow(final_data.astype(np.int32)[0])
plt.show()

plt.figure()
plt.imshow(x_test[0])
plt.show()

