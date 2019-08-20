from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

from Task05.Lenet import lenet


def train(mnist):
    # 训练数据 及 标签
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
    # 对数据进行训练
    y = lenet(x)

    # 交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.argmax(y_, 1))
    # 计算损失
    loss = tf.reduce_mean(cross_entropy)
    # 优化
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    #计算准确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(5001):
            xs, ys = mnist.train.next_batch(100)
            _, loss_value,acc = sess.run([train_op, loss,accuracy], feed_dict={x: xs, y_: ys})

            if i % 100 == 0:
                print("Step:",i," training batch loss:",loss_value," accuracy:",acc)

mnist = input_data.read_data_sets(r'MNIST_data', one_hot=True)
train(mnist)