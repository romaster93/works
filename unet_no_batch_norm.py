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

x_train_gray = (x_train_gray * 255).astype('int32')
x_test_gray = (x_test_gray * 255).astype('int32')


def xg_train(z):
    return np.reshape(x_train_gray[z], (32, 32))  # 49999개


def xg_test(z):
    return np.reshape(x_test_gray[z], (32, 32))


x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
y = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
n_batch = tf.placeholder(tf.int32)

##첫번째 레이어 쌓기
W_conv1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 64], stddev=5e-2))  # 32
h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME'))  # 32
W_conv2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=5e-2))  # 32
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME'))  # 32
#bn_conv2 = tf.layers.batch_normalization(h_conv2,momentum=0.9, training=True)

##MAX_POOLING
h_pool1 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # 16

##두번째 레이어 쌓기
W_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2))  # 16
h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv3, strides=[1, 1, 1, 1], padding='SAME'))  # 16
W_conv4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))  # 16
h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME'))  # 16
#bn_conv4 = tf.layers.batch_normalization(h_conv4,momentum=0.9, training=True)
##MAX_POOLING
h_pool2 = tf.nn.max_pool(h_conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # 8

##세번째 레이어 쌓기
W_conv5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], stddev=5e-2))  # 8
h_conv5 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv5, strides=[1, 1, 1, 1], padding='SAME'))  # 8
W_conv6 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], stddev=5e-2))  # 8
h_conv6 = tf.nn.relu(tf.nn.conv2d(h_conv5, W_conv6, strides=[1, 1, 1, 1], padding='SAME'))  # 8
#bn_conv6 = tf.layers.batch_normalization(h_conv6,momentum=0.9,
#                                         training=True)
##MAX_POOLING
h_pool3 = tf.nn.max_pool(h_conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # 4

##네번째 레이어 쌓기
W_conv7 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 512], stddev=5e-2))  # 4
h_conv7 = tf.nn.relu(tf.nn.conv2d(h_pool3, W_conv7, strides=[1, 1, 1, 1], padding='SAME'))  # 4
W_conv8 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=5e-2))  # 4
h_conv8 = tf.nn.relu(tf.nn.conv2d(h_conv7, W_conv8, strides=[1, 1, 1, 1], padding='SAME'))  # 4
#bn_conv8 = tf.layers.batch_normalization(h_conv8,momentum=0.9,
#                                         training=True)


##MAX_POOLING
h_pool4 = tf.nn.max_pool(h_conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # 2

##up conv 하기 직전
W_conv9 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 1024], stddev=5e-2))  # 2
h_conv9 = tf.nn.relu(tf.nn.conv2d(h_pool4, W_conv9, strides=[1, 1, 1, 1], padding='SAME'))  # 2
W_conv10 = tf.Variable(tf.truncated_normal(shape=[3, 3, 1024, 1024], stddev=5e-2))  # 2
h_conv10 = tf.nn.relu(tf.nn.conv2d(h_conv9, W_conv10, strides=[1, 1, 1, 1], padding='SAME'))  # 2
#bn_conv10 = tf.layers.batch_normalization(h_conv10,momentum=0.9,
#                                          training=True)


##upcov는 한칸만 해서 진행해보자
W_conv11 = tf.Variable(tf.truncated_normal(shape=[2, 2, 512, 1024], stddev=0.05))  # 2
h_conv11 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv10, W_conv11, output_shape=[n_batch, 4, 4, 512], strides=[1, 2, 2, 1]))  # 4
#bn_conv11 = tf.layers.batch_normalization(h_conv11, momentum=0.9
#    , training=True)
# output_shape = [?,4,4,512]
##잘라붙이기
cat_11 = tf.concat([h_conv8, h_conv11], axis=-1)

##컨벌루션 진행
W_convp1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 1024, 512], stddev=5e-2))
h_convp1 = tf.nn.relu(tf.nn.conv2d(cat_11, W_convp1, strides=[1, 1, 1, 1], padding='SAME'))
#bn_conv12 = tf.layers.batch_normalization(h_convp1,momentum=0.9,
#                                          training=True)

##upcov두번째층
W_conv12 = tf.Variable(tf.truncated_normal(shape=[2, 2, 256, 512], stddev=0.05))
h_conv12 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv11, W_conv12, output_shape=[n_batch, 32 // 4, 32 // 4, 256], strides=[1, 2, 2, 1],
                           padding='SAME'))
#bn_conv13 = tf.layers.batch_normalization(h_conv12
#    , momentum=0.9, training=True)

##잘라붙이기
cat_12 = tf.concat([h_conv6, h_conv12], axis=-1)

##컨벌루션 진행
W_convp2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 256], stddev=5e-2))
h_convp2 = tf.nn.relu(tf.nn.conv2d(cat_12, W_convp2, strides=[1, 1, 1, 1], padding='SAME'))
#bn_conv14 = tf.layers.batch_normalization(h_convp2,momentum=0.9,
#                                          training=True)

##upcov세번째층
W_conv13 = tf.Variable(tf.truncated_normal(shape=[2, 2, 128, 256], stddev=0.05))
h_conv13 = tf.nn.relu(tf.nn.conv2d_transpose(h_convp2, W_conv13, output_shape=[n_batch, 32 // 2, 32 // 2, 128], strides=[1, 2, 2, 1],
                           padding='SAME'))
#bn_conv15 = tf.layers.batch_normalization(h_conv13
#    , training=True)

##잘라붙이기
cat_13 = tf.concat([h_conv4, h_conv13], axis=-1)

##컨벌루션 진행
W_convp3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 128], stddev=5e-2))
h_convp3 = tf.nn.relu(tf.nn.conv2d(cat_13, W_convp3, strides=[1, 1, 1, 1], padding='SAME'))
#bn_conv16 = tf.layers.batch_normalization(h_convp3,momentum=0.9,
#                                          training=True)

##upcov네번째층
W_conv14 = tf.Variable(tf.truncated_normal(shape=[2, 2, 64, 128], stddev=0.05))
h_conv14 = tf.nn.relu(tf.nn.conv2d_transpose(h_convp3, W_conv14, output_shape=[n_batch, 32, 32, 64], strides=[1, 2, 2, 1],
                           padding='SAME'))
#bn_conv17 = tf.layers.batch_normalization(h_conv14
#    , momentum=0.9, training=True)

##잘라붙이기
cat_14 = tf.concat([h_conv2, h_conv14], axis=-1)

##컨벌루션 진행
W_convp4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 3], stddev=5e-2))
h_convp4 = tf.nn.relu(tf.nn.conv2d(cat_14, W_convp4, strides=[1, 1, 1, 1], padding='SAME'))  # 최종OUTPUT
#bn_conv18 = tf.layers.batch_normalization(h_convp4,momentum=0.9,
#                                          training=True)

##손실함수 활성화함수

loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=y, predictions=h_convp4))
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

# 정확도 계산하는거
# correct_prediction = tf.equal(tf.argmax(h_convp4,1 ), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

(x_train, y_train), (x_test, y_test) = load_data()

batch_size = 100
# 다 구성했으니까 시작한다.

# saver = tf.train.Saver()


sess = tf.Session()

# ckpt = tf.train.get_checkpoint_state('./')
# if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
#     saver.restore(sess, ckpt.model_checkpoint_path)
# else:
#   sess.run(tf.global_variables_initializer())

sess.run(tf.global_variables_initializer())

num_batch = int(len(x_train) / batch_size)
for epoch in range(500):
    for i in range(num_batch):
        start = i * batch_size
        end = start + batch_size
        _, cost, final_data = sess.run([train_step, loss, h_convp4],
                                       feed_dict={x: x_train_gray[start:end, :, :], y: x_train[start:end, :, :],
                                                  n_batch: batch_size})
    print(cost)

# saver.save(sess, 'my_test_model')


final_data = sess.run(h_convp4, feed_dict={x: x_test_gray[0].reshape(-1, 32, 32, 1), n_batch: 1})

plt.figure()
plt.imshow(final_data.astype(np.int32)[0])
plt.show()

plt.figure()
plt.imshow(x_test[0])
plt.show()

# test_accuracy = 0.0
    # for i in range(10):
    #     test_batch = next_batch(1000, x_test,y_test)
    #     test_accuracy = test_accuracy + accuracy.eval(feed_dict={x:test_batch[0], y:test_batch[1]})
    #
    # test_accuracy = test_accuracy / 10;
    # print("테스트 데이터에 대한 정확도 : %f" %test_accuracy)










