import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from sklearn.utils import shuffle
import  numpy as np
import os

from Task04.lenet import Lenet


def load_data():
   '''
   导入数据
   :return:训练集、测试集
   '''
   mnist = read_data_sets("MNIST_data/", reshape=False,one_hot=True)
   X_train, y_train = mnist.train.images, mnist.train.labels
   X_test, y_test = mnist.test.images, mnist.test.labels

   assert (len(X_train) == len(y_train))
   assert (len(X_test) == len(y_test))

   print()
   print("Image Shape: {}".format(X_train[0].shape))
   print()
   print("Training Set:   {} samples".format(len(X_train)))
   print("Test Set:       {} samples".format(len(X_test)))

   # 将训练集进行填充
   # 因为mnist数据集的图片是28*28*1的格式，而lenet只接受32*32的格式
   # 所以只能在这个基础上填充
   X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
   print("x_train_32:", X_train.shape)
   X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
   print("Updated Image Shape: {}".format(X_train[0].shape))

   # 打乱数据集的顺序
   X_train, y_train = shuffle(X_train, y_train)
   return X_train, y_train,X_test,y_test

def evaluate():
   num_examples = len(X_test)
   total_accuracy = 0
   sess = tf.get_default_session()  # 返回当前线程的默认会话
   for offset in range(0, num_examples, BATCH_SIZE):
      batch_x, batch_y = X_test[offset:offset + BATCH_SIZE], y_test[offset:offset + BATCH_SIZE]
      accuracy = sess.run(lenet5.accuracy, feed_dict={lenet5.x: batch_x, lenet5.y: batch_y})
      total_accuracy += (accuracy * len(batch_x))
   return total_accuracy / num_examples

BATCH_SIZE = 100
MODEL_SAVE_PATH = "model/"
MODEL_NAME="lenet.ckpt"
lenet5 = Lenet(0.001,0.1)
saver = tf.train.Saver()

init = tf.global_variables_initializer()
with tf.Session() as sess:
   sess.run(init)

   X_train, y_train,X_test,y_test= load_data()
   num_examples = len(X_train) #55000
   max_iter=int(num_examples/BATCH_SIZE)#迭代次数
   print("Start Training...")
   X_train, y_train = shuffle(X_train, y_train)  # 随机排序
   for j in range(max_iter):
         start=j*BATCH_SIZE
         end=start+BATCH_SIZE
         batch_xs,batch_ys=X_train[start:end],y_train[start:end]
         sess.run([lenet5.train_op],feed_dict={lenet5.x: batch_xs, lenet5.y: batch_ys})
         accuracy,loss=sess.run([lenet5.accuracy,lenet5.loss], feed_dict={lenet5.x: batch_xs, lenet5.y: batch_ys})
         print("Step:",j," accuracy:",accuracy," loss:",loss)
   saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
   # print("Model saved")

# Evaluate the Model
with tf.Session() as sess:
    saver.restore(sess,os.path.join(MODEL_SAVE_PATH, MODEL_NAME))  # 加载模型训练好的的网络和参数来测试，或进一步训练
    test_accuracy = evaluate()
    print("Test Accuracy = {:.3f}".format(test_accuracy))
