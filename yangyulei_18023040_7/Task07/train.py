import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import  numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import metrics
from Task07.lenet import  Lenet

mninst = input_data.read_data_sets('MNIST_data/', one_hot=True)

def createDataLib(batch_size,signal):
    #构建样本库
    full = 0  # 样本饱和
    dataLib_x = [[] for i in range(10)]  # 样本库
    # print(np.array(dataLib_x).shape) #(10, 0)
    dataLib_y = [[] for i in range(10)]

    while full != 10:
        if signal=="train":
           temp_x, temp_y = mninst.train.next_batch(batch_size)
        elif signal=="test":
           temp_x, temp_y = mninst.test.next_batch(batch_size)
        for i in range(0, batch_size):  # 将每行数据分类
            classNo = np.argmax(temp_y[i])

            if len(dataLib_x[classNo]) == batch_size:  # 样本库中每个类别数据存batch_size个
                continue

            if len(dataLib_x[classNo]) == batch_size - 1:
                dataLib_x[classNo].append(temp_x[i])
                dataLib_y[classNo].append(temp_y[i])
                full += 1
                continue

            dataLib_x[classNo].append(temp_x[i])
            dataLib_y[classNo].append(temp_y[i])
    return dataLib_x,dataLib_y

def getTrainDatas(batch_size):
    input_x1 = []
    input_x2 = []
    input_y1 = []
    input_y2 = []

    dataLib_x,dataLib_y=createDataLib(batch_size,"train")
    # 取得正例
    for i in range(0, 10):
        for j in range(0, 450):#每个种类的正样本
            randomNumber1 = random.randint(0, batch_size - 1)
            input_x1.append(dataLib_x[i][randomNumber1]) #i表示类别
            input_y1.append(dataLib_y[i][randomNumber1])
            randomNumber2 = random.randint(0, batch_size - 1)
            input_x2.append(dataLib_x[i][randomNumber2])
            input_y2.append(dataLib_y[i][randomNumber2])

    # 取得反例
    for i in range(0, 9):
        for j in range(i + 1, 10):
            for k in range(0, 100):#每个种类的负样本append100个
                randomNumber1 = random.randint(0, batch_size - 1)
                input_x1.append(dataLib_x[i][randomNumber1])
                input_y1.append(dataLib_y[i][randomNumber1])
                randomNumber2 = random.randint(0, batch_size - 1)
                input_x2.append(dataLib_x[j][randomNumber2])
                input_y2.append(dataLib_y[j][randomNumber2])

    input_x1 = np.array(input_x1).reshape((-1, 28, 28, 1))
    input_x2 = np.array(input_x2).reshape((-1, 28, 28, 1))
    input_y1 = np.array(input_y1).reshape((-1, 10))
    input_y2 = np.array(input_y2).reshape((-1, 10))

    # print(input_x1.shape)  #(9000, 28, 28, 1)
    return input_x1, input_x2, input_y1, input_y2


def getTestTatas(batch_size):
    # 组装构造例子
        input_x1 = []
        input_x2 = []
        input_y1 = []
        input_y2 = []

        dataLib_x,dataLib_y=createDataLib(batch_size,"test")

        # 取得正例
        for i in range(0, 10):
            for j in range(0, 450):
                randomNumber1 = random.randint(0, batch_size - 1)
                input_x1.append(dataLib_x[i][randomNumber1])
                input_y1.append(dataLib_y[i][randomNumber1])
                randomNumber2 = random.randint(0, batch_size - 1)
                input_x2.append(dataLib_x[i][randomNumber2])
                input_y2.append(dataLib_y[i][randomNumber2])
            print('Positive (%d,%d):' % (i, i), '%d' % (450))
        print('Pos Total:%d' % (4500))

        # 取得反例
        for i in range(0, 9):
            for j in range(i + 1, 10):
                for k in range(0, 100):
                    randomNumber1 = random.randint(0, batch_size - 1) #[0,batch_size - 1]中的一个
                    input_x1.append(dataLib_x[i][randomNumber1])
                    input_y1.append(dataLib_y[i][randomNumber1])
                    randomNumber2 = random.randint(0, batch_size - 1)
                    input_x2.append(dataLib_x[j][randomNumber2])
                    input_y2.append(dataLib_y[j][randomNumber2])
                print('Positive (%d,%d):' % (i, j), '%d' % (100))

        print('Neg Total:%d' % (4500))

        print('Total:%d' % (9000))

        input_x1 = np.array(input_x1).reshape((-1, 28, 28, 1))
        input_x2 = np.array(input_x2).reshape((-1, 28, 28, 1))
        input_y1 = np.array(input_y1).reshape((-1, 10))
        input_y2 = np.array(input_y2).reshape((-1, 10))

        return input_x1, input_x2, input_y1, input_y2

def main():
    LOGDIR="tensorboard"
    '''
    0.001：learning_rate
    0.1：sigma
    0：mu
    0.2：threshold
    15：Q
    '''
    lenet= Lenet(0.001,0.1,0,0.2,15)
    merged_summary = tf.summary.merge_all()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(LOGDIR + "/")
        writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer(),sess.run(tf.local_variables_initializer()))
        #训练
        print("Start Training:")
        for i in range(20000):
            input_x1, input_x2, input_y1, input_y2 = getTrainDatas(1024)
            if i % 50 == 0:
                s = sess.run(merged_summary, feed_dict={
                    lenet.x1: input_x1,
                    lenet.x2: input_x2,
                    lenet.y1: input_y1,
                    lenet.y2: input_y2
                })
                writer.add_summary(s, i)
            _, loss, label, acc,ew = sess.run([lenet.train_op,
                                               lenet.loss,
                                               lenet.label,
                                               lenet.accuracy,
                                               lenet.ew,
                                               ],
                                              {
                                                  lenet.x1: input_x1,
                                                  lenet.x2: input_x2,
                                                  lenet.y1: input_y1,
                                                  lenet.y2: input_y2
                                              })
            precision, recall, _thresholds = metrics.precision_recall_curve(label, ew)
            # acc = _compute_acc(label, y_prediction)
            auc = metrics.auc(recall, precision)

            if i % 5==0:
              print('setp:%d' % i, 'loss:', loss, ' ', 'auc', auc,"acc",acc)
            if acc > 0.95:
                break

        #测试
        input_x1, input_x2, input_y1, input_y2 = getTestTatas(1024)
        label, acc,ew, = sess.run([lenet.label,
                               lenet.accuracy,
                               lenet.ew],
                                  {
                                      lenet.x1: input_x1,
                                      lenet.x2: input_x2,
                                      lenet.y1: input_y1,
                                      lenet.y2: input_y2
                                  })
        precision, recall, _thresholds = metrics.precision_recall_curve(label, ew)
        print("在测试集上的准确率：",acc)
        auc = metrics.auc(recall, precision)
        print(auc)
        plt.plot(recall, precision)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.savefig("pr.jpg")
        plt.show()
        #AP：PR曲线与X轴围成的图形面积
        # AUC：计算出ROC曲线下面的面


if __name__ == '__main__':
    main()
