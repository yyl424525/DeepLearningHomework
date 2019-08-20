import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from sklearn.utils import shuffle
import  numpy as np
import matplotlib.pyplot as plt
import random
import os

from Task04.lenet2 import Lenet2


def load_data():
   '''
   导入数据
   :return:训练集、验证集、测试集
   '''
   #    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
   mnist = read_data_sets("MNIST_data/", reshape=False,one_hot=True)
   X_train, y_train = mnist.train.images, mnist.train.labels
   X_validation, y_validation = mnist.validation.images, mnist.validation.labels
   X_test, y_test = mnist.test.images, mnist.test.labels

   assert (len(X_train) == len(y_train))
   assert (len(X_validation) == len(y_validation))
   assert (len(X_test) == len(y_test))

   print()
   print("Image Shape: {}".format(X_train[0].shape))
   print()
   print("Training Set:   {} samples".format(len(X_train)))
   print("Validation Set: {} samples".format(len(X_validation)))
   print("Test Set:       {} samples".format(len(X_test)))

   # 打乱数据集的顺序
   # X_train, y_train = shuffle(X_train, y_train)
   return X_train, y_train,X_test,y_test

def evaluate():
   num_examples = len(X_test)
   total_accuracy = 0
   sess = tf.get_default_session()  # 返回当前线程的默认会话
   for offset in range(0, num_examples, BATCH_SIZE):
      batch_x, batch_y = X_test[offset:offset + BATCH_SIZE], y_test[offset:offset + BATCH_SIZE]
      accuracy = sess.run(lenet5_2.accuracy, feed_dict={lenet5_2.x: batch_x, lenet5_2.y: batch_y})
      total_accuracy += (accuracy * len(batch_x))
   return total_accuracy / num_examples

BATCH_SIZE = 100
MODEL_SAVE_PATH = "model2/"
MODEL_NAME="lenet2.ckpt"
lenet5_2 = Lenet2(0.001,0.1)
saver = tf.train.Saver()

init = tf.global_variables_initializer()
with tf.Session() as sess:
   sess.run(init)

   X_train, y_train,X_test,y_test= load_data()
   print("y_train:", y_train.shape)
   num_examples = len(X_train) #55000
   max_iter=int(num_examples/BATCH_SIZE)#迭代次数
   print("Start Training...")
   # X_train, y_train = shuffle(X_train, y_train)  # 随机排序
   for j in range(max_iter):
         start=j*BATCH_SIZE
         end=start+BATCH_SIZE
         batch_xs,batch_ys=X_train[start:end],y_train[start:end]
         sess.run([lenet5_2.train_op],feed_dict={lenet5_2.x: batch_xs, lenet5_2.y: batch_ys})
         accuracy,loss=sess.run([lenet5_2.accuracy,lenet5_2.loss], feed_dict={lenet5_2.x: batch_xs, lenet5_2.y: batch_ys})
         print("Step:",j," accuracy:",accuracy," loss:",loss)
   saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
   # print("Model saved")

# Evaluate the Model
with tf.Session() as sess:
    saver.restore(sess,os.path.join(MODEL_SAVE_PATH, MODEL_NAME))  # 加载模型训练好的的网络和参数来测试，或进一步训练
    test_accuracy = evaluate()
    print("Test Accuracy = {:.3f}".format(test_accuracy))
