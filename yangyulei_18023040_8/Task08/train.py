import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import  numpy as np
import os
import random
import matplotlib.pyplot as plt
from Task08.lenet import Lenet

mninst = read_data_sets("MNIST_data/", reshape=False, one_hot=True)

def load_data(signal,batch):
   '''
   导入数据
   :return:训练集、测试集
   '''
   X_train, y_train = mninst.train.next_batch(batch)
   X_valid,y_valid=mninst.validation.next_batch(batch)
   X_test, y_test = mninst.test.next_batch(batch)

   assert (len(X_train) == len(y_train))
   assert (len(X_test) == len(y_test))


   # 将训练集进行填充
   # 因为mnist数据集的图片是28*28*1的格式，而lenet只接受32*32的格式
   # 所以只能在这个基础上填充
   X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
   X_valid = np.pad(X_valid, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
   X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

   if signal=="train":
     return X_train, y_train
   if signal=="validation":
     return X_valid,y_valid
   elif signal=="test":
     return X_test, y_test


batch_size = 8
LOGDIR="tensorboard"
iter = 1000 # 迭代次数


def main():
    lenet5 = Lenet(0.001, 0.1)
    merged_summary = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    sess=tf.Session()

    writer = tf.summary.FileWriter(LOGDIR + "/" )

    writer.add_graph(sess.graph)
    sess.run(init)

    #开始训练
    print("Start Training...")
    for j in range(iter):
         X_train, y_train=load_data("train",batch_size)
         _, accuracy,loss=sess.run([lenet5.train_op,lenet5.accuracy,lenet5.loss], feed_dict={lenet5.x: X_train, lenet5.y: y_train})
         if j % 10 == 0:
             X_valid,y_valid=load_data("validation",100)
             s,accuracy,loss = sess.run([merged_summary,lenet5.accuracy,lenet5.loss], feed_dict={lenet5.x: X_valid, lenet5.y: y_valid})
             writer.add_summary(s, j)
             print("Step:",j," validation accuracy:",accuracy," loss:",loss)

if __name__ == '__main__':
    main()