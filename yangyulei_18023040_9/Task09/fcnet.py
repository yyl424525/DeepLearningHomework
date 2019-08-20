import tensorflow as tf
import numpy as np


class fcNet:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        # 在创建的时候运行画图
        self._build_graph()

    # 涉及网络的所有画图build graph过程,常用一个build graph封起来
    def _build_graph(self, network_name='fcNet'):
        self._setup_placeholders_graph()
        self._build_network_graph(network_name)
        self._compute_loss_graph()
        self._create_train_op_graph()
        self._compute_acc_graph()

    # 构建图
    def _setup_placeholders_graph(self):
        # self.x = tf.placeholder(tf.float32, (None, 32, 32, 1), name='input_x')
        self.x = tf.placeholder(tf.float32, [None, 784], name='input_x')
        self.y = tf.placeholder(tf.int32, [None,10])  # 在模型中的占位

    def _build_network_graph(self, scope_name):
        with tf.variable_scope(scope_name):
            #生成一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0, 1)，不包括1
            w1 = tf.Variable(dtype=tf.float32, initial_value=np.random.rand(784, 1500), name='w1') #维度(784, 1500)
            w2 = tf.Variable(dtype=tf.float32, initial_value=np.random.rand(1500, 1000), name='w2')
            w3 = tf.Variable(dtype=tf.float32, initial_value=np.random.rand(1000, 500), name='w3')
            w4 = tf.Variable(dtype=tf.float32, initial_value=np.random.rand(500, 10), name='w4')
            b1 = tf.Variable(dtype=tf.float32, initial_value=np.random.rand(1500), name='b1') #返回一个值
            b2 = tf.Variable(dtype=tf.float32, initial_value=np.random.rand(1000), name='b2')
            b3 = tf.Variable(dtype=tf.float32, initial_value=np.random.rand(500), name='b3')
            b4 = tf.Variable(dtype=tf.float32, initial_value=np.random.rand(10), name='b4')

            fc1 = tf.nn.relu(tf.matmul(self.x, w1) + b1)
            fc2 = tf.nn.relu(tf.matmul(fc1, w2) + b2)
            fc3 = tf.nn.relu(tf.matmul(fc2, w3) + b3)
            fc4 = tf.matmul(fc3, w4) + b4

            self.digits = fc4
            return self.digits

    def _compute_loss_graph(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.digits)
        self.loss = tf.reduce_mean(cross_entropy)
        #下面代码为参数使用正则化
        # tv = tf.trainable_variables()
        # lambda_l = 0.0005
        # Regularization_term = lambda_l * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
        # self.loss = Regularization_term + self.loss
        # tf.summary.scalar("loss", self.loss)

    def _create_train_op_graph(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _compute_acc_graph(self):
        # calculate correct
        self.prediction = tf.equal(tf.argmax(self.digits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.prediction, tf.float32))
        tf.summary.scalar("accuracy", self.accuracy)
