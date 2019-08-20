import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np
from Task09.fcnet import fcNet
import matplotlib.pyplot as plt
from matplotlib import image
import tensorflow.contrib.slim as slim

#reshape=False，X_train.shape=(28,28)
#默认reshape=True，X_train.shape=(,784)
mninst = read_data_sets("MNIST_data/",  one_hot=True)


def load_data(signal, batchsize):
    '''
    导入数据
    :return:训练集、测试集、验证集
    '''
    X_train, y_train = mninst.train.next_batch(batchsize)
    X_valid = mninst.validation.images
    y_valid = mninst.validation.labels
    X_test = mninst.test.images
    y_test = mninst.test.labels

    assert (len(X_train) == len(y_train))
    assert (len(X_test) == len(y_test))

    if signal == "train":
        return X_train, y_train
    if signal == "validation":
        return X_valid, y_valid
    elif signal == "test":
        return X_test, y_test

def img_save(learning_rate,dict):
    x= [i*100 for i in range(0, 300)]
    for k,v in dict.items():
        print(k)
        print(v)
        p1,=plt.plot(x, v, label=k)
        plt.xlabel("iteration")
        plt.legend([p1],[k], loc='upper left')
        name = str(learning_rate) + k + ".png"
        plt.savefig(name)
        plt.close()

def save_w(learning_rate,w1,w2,w3,w4):
    w1_min = np.min(w1)
    w1_max = np.max(w1)
    w1_0_to_1 = (w1 - w1_min) / (w1_max - w1_min)
    image.imsave(str(learning_rate)+'w1.png', w1_0_to_1)

    w2_min = np.min(w2)
    w2_max = np.max(w2)
    w2_0_to_1 = (w2 - w2_min) / (w2_max - w2_min)
    image.imsave(str(learning_rate)+'w2.png', w2_0_to_1)

    w3_min = np.min(w3)
    w3_max = np.max(w3)
    w3_0_to_1 = (w3 - w3_min) / (w3_max - w3_min)
    image.imsave(str(learning_rate)+'w3.png', w3_0_to_1)

    w4_min = np.min(w4)
    w4_max = np.max(w4)
    w4_0_to_1 = (w4 - w4_min) / (w4_max - w4_min)
    image.imsave(str(learning_rate)+'w4.png', w4_0_to_1)

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

TRAINING_STEPS=30000
batchsize=64
def main():
    for learning_rate in [0.0001,0.005]:
        training_loss=[]
        validation_loss=[]
        vaildation_acc=[]
        test_acc=0
        fcnet=fcNet(learning_rate)
        # 显示所有变量
        show_all_variables()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for step in range(TRAINING_STEPS):
               X_train,y_train=load_data("train",batchsize)
               _,train_loss,train_acc=sess.run([fcnet.train_op,fcnet.loss,fcnet.accuracy],feed_dict={fcnet.x:X_train,fcnet.y:y_train})

               if  step%100==99:
                   X_valid, y_valid = load_data("validation", batchsize)
                   valid_loss, valid_acc = sess.run([fcnet.loss, fcnet.accuracy],
                                                    feed_dict={fcnet.x: X_valid, fcnet.y: y_valid})

                   X_test, y_test = load_data("test", batchsize)
                   training_loss.append(train_loss)
                   validation_loss.append(valid_loss)
                   vaildation_acc.append(valid_acc)
                   print("step=",step,",train_loss=",train_loss,",valid_loss=",valid_loss,",valid_acc=",valid_acc)
            test_loss, test_acc = sess.run([fcnet.loss, fcnet.accuracy],feed_dict={fcnet.x: X_test, fcnet.y: y_test})
            #输出最终的test_acc
            print("test acc:",test_acc)

            #获取图里的所有tensor
            # graph = tf.get_default_graph()
            # for op in graph.get_operations():
            #     print(op.name)

            w1 = sess.run('fcNet/w1:0')
            w2= sess.run('fcNet/w2:0')
            w3 = sess.run('fcNet/w3:0')
            w4 = sess.run('fcNet/w4:0')
            #方法二
            # w1 = sess.run(graph.get_tensor_by_name("fcNet/w1:0"))
            save_w(learning_rate,w1,w2,w3,w4)
            img_save(learning_rate,{"train_loss":training_loss,"valid_loss":validation_loss,"valid_acc":vaildation_acc})

if __name__ == '__main__':
    main()
