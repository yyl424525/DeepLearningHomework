import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import  numpy as np
import os
import random
import matplotlib.pyplot as plt
from Task06.lenet import Lenet


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
   return X_train, y_train,X_test,y_test

def load_test(n):
    # 不用onehot读取，这样label就是0-9的数
    mnist1 = read_data_sets("MNIST_data/")
    # labels:[7 3 4 ..., 5 6 8]
    labels= mnist1.test.labels

    # 一个全是n的一维数组:[ n.  n.  n. ...,  n.  n.  n.]
    nArray = n * np.ones(labels.shape)
    # 对比得到一个模板，用来筛选数字为n的图片:mask:[False  True False ..., False False False]
    mask = np.equal(labels, nArray)

    mnist2 = read_data_sets("MNIST_data/",reshape=False,one_hot=True)
    #取出相同数字的不同测试样本图片
    X_test=mnist2.test.images[mask, :]
    X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

    y_test=mnist2.test.labels[mask, :]
    #取一百行
    X=X_test[:100]
    y=y_test[:100]
    #(100, 784)
    print("x.shape",X.shape)
    return X,y



BATCH_SIZE = 100
MODEL_SAVE_PATH = "model/"
MODEL_NAME="lenet.ckpt"
LOGDIR="tensorboard"
# ,"sigmoid"
for activation_function in ["relu","sigmoid"]:
    tf.reset_default_graph()

    lenet5 = Lenet(0.001, 0.1,activation_function)
    saver = tf.train.Saver()
    merged_summary = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    sess=tf.Session()

    writer = tf.summary.FileWriter(LOGDIR + "/" + activation_function)

    writer.add_graph(sess.graph)
    sess.run(init)

    X_train, y_train, X_test, y_test = load_data()
    num_examples = len(X_train) #55000
    max_iter=int(num_examples/BATCH_SIZE)#迭代次数
    print("Start Training...")
    for j in range(max_iter):
         start=j*BATCH_SIZE
         end=start+BATCH_SIZE
         batch_xs,batch_ys=X_train[start:end],y_train[start:end]
         if j % 5 == 0:
             s = sess.run(merged_summary, feed_dict={lenet5.x: batch_xs, lenet5.y: batch_ys})
             writer.add_summary(s, j)
         sess.run([lenet5.train_op],feed_dict={lenet5.x: batch_xs, lenet5.y: batch_ys})
         accuracy,loss=sess.run([lenet5.accuracy,lenet5.loss], feed_dict={lenet5.x: batch_xs, lenet5.y: batch_ys})
         print("Step:",j," accuracy:",accuracy," loss:",loss)
    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
    print("Model saved")


    # # Evaluate the Model
    # saver.restore(sess,os.path.join(MODEL_SAVE_PATH, MODEL_NAME))  # 加载模型训练好的的网络和参数来测试，或进一步训练
    #
    # num_examples = len(X_test)
    # total_accuracy = 0
    # for offset in range(0, num_examples, BATCH_SIZE):
    #     batch_x, batch_y = X_test[offset:offset + BATCH_SIZE], y_test[offset:offset + BATCH_SIZE]
    #     accuracy,fc2= sess.run([lenet5.accuracy,lenet5.fc2_relu], feed_dict={lenet5.x: batch_x, lenet5.y:batch_y})
    #     total_accuracy += (accuracy * len(batch_x))
    # test_accuracy=total_accuracy / num_examples
    # print("Test Accuracy = {:.3f}".format(test_accuracy))


    with tf.Session() as sess:

      # Evaluate the Model
      saver.restore(sess,os.path.join(MODEL_SAVE_PATH, MODEL_NAME))  # 加载模型训练好的的网络和参数来测试，或进一步训练
      sess = tf.get_default_session()  # 返回当前线程的默认会话
      total_accuracy = 0

      images={i:[] for i in range(10)}
      for i in range(10):
        X_test, y_test = load_test(i)
        # X_test, y_test = load_test(i)
        fc2= sess.run([lenet5.fc2_relu], feed_dict={lenet5.x: X_test, lenet5.y: y_test})
        # print("Test Accuracy = {:.3f}".format(test_accuracy))
        fc2=(fc2-np.min(fc2))/(np.max(fc2)-np.min(fc2))
        print("fc2.shape",fc2.shape)
        fc2=np.reshape(fc2,[100,84])
        images[i]=fc2
      for i in range(10):
        #一行10列
        plt.subplot(1,10,i+1)
        plt.imshow(images[i],cmap='gray')
        plt.title(i)
        plt.axis('off')
      plt.savefig(activation_function+'_test.png')
      plt.show()

