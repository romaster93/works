import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tensorflow.keras.datasets.cifar10 import load_data

(x_train, y_train), (x_test, y_test) = load_data()

def next_batch(num, data, labels):

    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle=[labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)



x = tf.placeholder(tf.float32, shape =[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape =[None, 32, 32, 3])
n_batch = tf.placeholder(tf.int32)

##첫번째 레이어 쌓기
W_conv1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], stddev=5e-2))                        #32
h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME'))                #32
W_conv2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=5e-2))                       #32
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME'))          #32
##MAX_POOLING
h_pool1 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")               #16

##두번째 레이어 쌓기
W_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2))                      #16
h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv3, strides=[1, 1, 1, 1], padding='SAME'))          #16
W_conv4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))                     #16
h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME'))          #16
##MAX_POOLING
h_pool2 = tf.nn.max_pool(h_conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")               #8

##세번째 레이어 쌓기
W_conv5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], stddev=5e-2))                     #8
h_conv5 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv5, strides=[1, 1, 1, 1], padding='SAME'))          #8
W_conv6 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], stddev=5e-2))                     #8
h_conv6 = tf.nn.relu(tf.nn.conv2d(h_conv5, W_conv6, strides=[1, 1, 1, 1], padding='SAME'))          #8
##MAX_POOLING
h_pool3 = tf.nn.max_pool(h_conv6, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")               #4

##네번째 레이어 쌓기
W_conv7 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 512], stddev=5e-2))                     #4
h_conv7 = tf.nn.relu(tf.nn.conv2d(h_pool3, W_conv7, strides=[1, 1, 1, 1], padding='SAME'))          #4
W_conv8 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=5e-2))                     #4
h_conv8 = tf.nn.relu(tf.nn.conv2d(h_conv7, W_conv8, strides=[1, 1, 1, 1], padding='SAME'))          #4
##MAX_POOLING
h_pool4 = tf.nn.max_pool(h_conv8, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")               #2

##up conv 하기 직전
W_conv9 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 1024], stddev=5e-2))                    #2
h_conv9 = tf.nn.relu(tf.nn.conv2d(h_pool4, W_conv9, strides=[1, 1, 1, 1], padding='SAME'))          #2
W_conv10 = tf.Variable(tf.truncated_normal(shape=[3, 3, 1024, 1024], stddev=5e-2))                  #2
h_conv10 = tf.nn.relu(tf.nn.conv2d(h_conv9, W_conv10, strides=[1, 1, 1, 1], padding='SAME'))        #2
print(h_conv10)

##upcov는 한칸만 해서 진행해보자
W_conv11 = tf.Variable(tf.truncated_normal(shape = [2,2,512,1024], stddev=0.05))                    #2
h_conv11 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv10, W_conv11, output_shape=[n_batch,4,4,512], strides=[1, 2, 2, 1], padding='SAME'))      #4
# output_shape = [?,4,4,512]
##잘라붙이기
cat_11 = tf.concat([h_conv8, h_conv11], axis = -1)

print(cat_11)

##컨벌루션 진행
W_convp1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 1024, 512], stddev=5e-2))
h_convp1 = tf.nn.relu(tf.nn.conv2d(cat_11, W_convp1, strides=[1, 1, 1, 1], padding='SAME'))

##upcov두번째층
W_conv12 = tf.Variable(tf.truncated_normal(shape = [2,2,256,512], stddev=0.05))
h_conv12 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv11, W_conv12, output_shape = [n_batch,32//4,32//4,256], strides=[1, 2, 2, 1], padding='SAME'))



##잘라붙이기
cat_12 = tf.concat([h_conv6, h_conv12], axis = -1)

##컨벌루션 진행
W_convp2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 256], stddev=5e-2))
h_convp2 = tf.nn.relu(tf.nn.conv2d(cat_12, W_convp2, strides=[1, 1, 1, 1], padding='SAME'))

##upcov세번째층
W_conv13 = tf.Variable(tf.truncated_normal(shape = [2,2,128,256], stddev=0.05))
h_conv13 = tf.nn.relu(tf.nn.conv2d_transpose(h_convp2, W_conv13, output_shape = [n_batch,32//2,32//2,128], strides=[1, 2, 2, 1], padding='SAME'))

##잘라붙이기
cat_13 = tf.concat([h_conv4, h_conv13], axis = -1)
print(cat_13)

##컨벌루션 진행
W_convp3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 128], stddev=5e-2))
h_convp3 = tf.nn.relu(tf.nn.conv2d(cat_13, W_convp3, strides=[1, 1, 1, 1], padding='SAME'))


##upcov네번째층
W_conv14 = tf.Variable(tf.truncated_normal(shape = [2,2,64,128], stddev=0.05))
h_conv14 = tf.nn.relu(tf.nn.conv2d_transpose(h_convp3, W_conv14, output_shape = [n_batch,32,32,64], strides=[1, 2, 2, 1], padding='SAME'))

##잘라붙이기
cat_14 = tf.concat([h_conv2, h_conv14], axis = -1)

print(cat_14)

##컨벌루션 진행
W_convp4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 3], stddev=5e-2))
h_convp4 = tf.nn.relu(tf.nn.conv2d(cat_14, W_convp4, strides=[1, 1, 1, 1], padding='SAME'))

##손실함수 활성화함수

# loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=x, logits=h_convp4))
loss = tf.reduce_mean(tf.losses.mean_squared_error(labels = y, predictions = h_convp4))
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

#정확도 계산하는거
# correct_prediction = tf.equal(tf.argmax(h_convp4,1 ), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

(x_train, y_train), (x_test, y_test) = load_data()

batch_size = 1000
#다 구성했으니까 시작한다.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    num_batch = int(len(x_train)/batch_size)
    for epoch in range(1000):
        for i in range(num_batch):
            start = i * batch_size
            end = start + batch_size
            _, cost = sess.run([train_step, loss], feed_dict={x: x_train[start:end,:,:], y: x_train[start:end,:,:], n_batch: batch_size})
        print(cost)


        # if i %100 == 0:
            # train_accuracy = accuracy.eval(feed_dict={x:batch[0], y:batch[0]})
            # loss_print = loss.eval(feed_dict={x:batch[0], y:batch[0]})

            # print("반복(epoch) : %d, 트레이닝 데이터 정확도 : %f, 손실 함수(loss): %f " %(i, train_accuracy, loss_print))



    # test_accuracy = 0.0
    # for i in range(10):
    #     test_batch = next_batch(1000, x_test,y_test)
    #     test_accuracy = test_accuracy + accuracy.eval(feed_dict={x:test_batch[0], y:test_batch[1]})
    #
    # test_accuracy = test_accuracy / 10;
    # print("테스트 데이터에 대한 정확도 : %f" %test_accuracy)










