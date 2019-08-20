import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import flatten


class Lenet:
    '''
    threshold:根据距离判断是否是同一数字的阈值
    '''
    def __init__(self, learning_rate, sigma, mu,threshold,Q):
        self.learning_rate = learning_rate
        self.Q = Q
        self.sigma = sigma
        self.mu = mu
        self.threshold=threshold
        # 当 在 创 建 的 时 候 运 行 画 图
        self._build_graph()

    # 涉及网络的所有画图build graph过程,常用一个build graph封起来
    def _build_graph(self, network_name='Lenet'):
        self._setup_placeholders_graph()
        self._compute_feature_ew()
        self._get_label()
        self._compute_loss_graph()
        self._compute_y_prediction()
        self._create_train_op_graph()
        self._compute_acc_graph() #不加入会出现AttributeError: 'Lenet' object has no attribute 'accuracy'的错误
        self.merged_summary = tf.summary.merge_all()

    def _cnn_layer(self, scope_name, W_name, b_name, x, filter_shape, conv_strides, b_shape, padding_tag='VALID'):
        print("input_x_shape:", x)
        with tf.variable_scope(scope_name) as scope:
            conv_weights = tf.get_variable(name=W_name, shape=filter_shape,
                                           initializer=tf.truncated_normal_initializer(mean=self.mu, stddev=self.sigma))
            conv_biases = tf.get_variable(name=b_name, shape=b_shape, initializer=tf.constant_initializer(0.0))
            # 使用边长为5，深度为32的过滤器，过滤器移动的步长为1，且使用全0填充
            print("x_shape:", x.shape, "  conv1_w_shape:", conv_weights.shape)
            conv = tf.nn.conv2d(x, conv_weights, strides=conv_strides, padding=padding_tag)
            return tf.nn.relu(tf.nn.bias_add(conv, conv_biases))


    def _pooling_layer(self, scope_name, relu, pool_ksize, pool_strides, padding_tag='VALID'):
        with tf.variable_scope(scope_name) as scope:
            return tf.nn.max_pool(relu, ksize=pool_ksize, strides=pool_strides, padding=padding_tag)

    def _flatten(self, pool2):
        # 将x拉直
        pool_shape = pool2.get_shape().as_list()
        length = pool_shape[1] * pool_shape[2] * pool_shape[3]
        return tf.reshape(pool2, [pool_shape[0], length])

    def _fully_connected_layer(self, scope_name, W_name, b_name, x, W_shape, b_shape):
        with tf.variable_scope(scope_name) as scope:
            fc_weights = tf.get_variable(W_name, W_shape,
                                         initializer=tf.truncated_normal_initializer(mean=self.mu, stddev=self.sigma))
            fc_biases = tf.get_variable(b_name, b_shape, initializer=tf.constant_initializer(0.0))
            return tf.nn.relu(tf.matmul(x, fc_weights) + fc_biases)
        # tf.matmul(x,y):x和y需要满足矩阵乘法要求

    # 构建图
    def _setup_placeholders_graph(self):
        # self.x  = tf.placeholder(tf.float32, (None, 28, 28, 1),name='input_x')
        # print(self.x.shape)
        # self.y= tf.placeholder(tf.int32, (None))  # 在模型中的占位

        # x1和x2中对应的前4500是正样本，即y相同；后4500是负样本，y不同
        self.x1 = tf.placeholder("float", shape=[None, 28, 28, 1], name='x1')
        self.x2 = tf.placeholder("float", shape=[None, 28, 28, 1], name='x2')
        self.y1 = tf.placeholder("float", shape=[None, 10], name='y1')
        self.y2 = tf.placeholder("float", shape=[None, 10], name='y2')

    def _get_feature(self, scope_name,x):
        with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE):
            # 第一个卷积层
            # Input = 32x32x1. Output = 28x28x6.
            # 卷积核：[filter_height, filter_width, in_channels, out_channels]
            self.conv1_relu = self._cnn_layer('layer_1_conv', 'conv1_w', 'conv1_b',x, (5, 5, 1, 6), [1, 1, 1, 1],[6])

            # 第一个池化层
            # Input = 28x28x6. Output = 14x14x6.
            # 图像序列 x 高 x 宽 x 通道序列;步长只设定在“高”和“宽”的维度为 2。
            self.pool1 = self._pooling_layer('layer_2_pooling', self.conv1_relu, [1, 2, 2, 1], [1, 2, 2, 1])

            # 第二个卷积层
            # Output = 10x10x16.

            self.conv2_relu = self._cnn_layer('layer_3_conv', 'conv2_w', 'conv2_b', self.pool1, (5, 5, 6, 16),
                                              [1, 1, 1, 1], [16])

            # 第二个池化层
            # Input = 10x10x16. Output = 5x5x16.
            self.pool2 = self._pooling_layer('layer_4_pooling', self.conv2_relu, [1, 2, 2, 1], [1, 2, 2, 1])

            # Tensor to vector:输入维度由 Nx5x5x16 压平后变为 Nx400
            print("self.pool2.shape:", self.pool2.shape)
            self.pool2 = flatten(self.pool2)

            # 第一个全连接层
            # Input = 256. Output = 120.
            self.fc1_relu = self._fully_connected_layer('layer_5_fc1', 'fc1_w', 'fc1_b', self.pool2, (256, 120), [120])

            # 第二个全连接层
            # Input = 120. Output = 84.
            self.fc2_relu = self._fully_connected_layer('layer_6_fc2', 'fc2_w', 'fc2_b', self.fc1_relu, (120, 84), [84])

            # 第三个全连接层
            # Input = 84. Output = 10.
            self.fc3_relu = self._fully_connected_layer('layer_7_fc3', 'fc3_w', 'fc3_b', self.fc2_relu, (84, 10), [10])

            self.digits = self.fc3_relu  # 100*10

            return self.digits

    #计算欧式距离，最后加上一个1e-6防止梯度消失
    def _compute_feature_ew(self):
        f1 = self._get_feature('x1_getfeature', self.x1) #f1.shape: (?, 10)
        f2 = self._get_feature('x2_getfeature', self.x2)
        self.ew= tf.sqrt(tf.reduce_sum(tf.square(f1 - f2), 1) + 1e-6)
        print("self.feature_distance.shape:",self.ew)

    #计算预测标签值
    def _compute_y_prediction(self):
        with tf.variable_scope('predict_label'):
            #将距离转化在0到1之间，距离越小越相似
            self.predict_label = self.ew / tf.reduce_max(self.ew)
            ones=tf.ones_like(self.label)
            zeros=tf.zeros_like(self.label)
            #设置距离大于自定义的threshold时为1，小于为0，即距离为0的为相同数字的图片
            self.predict_label=tf.where(self.predict_label<self.threshold,x=zeros,y=ones)

    #计算真实标签值
    def _get_label(self):
        #label:[F,F,F,F,F,F,...,T,T,T,T,T,T],然后转换成[0,0,0,0，...，1,1，1]
        #即此处标签值为0则表示两图片相同，为1则不同
        self.label = tf.cast(tf.not_equal(tf.argmax(self.y1, axis=1), tf.argmax(self.y2, axis=1)), dtype=tf.float32)

    def _compute_loss_graph(self):
        with tf.name_scope("loss_function"):
            #相同为0，不同为1
            #采用作业三中的损失函数：
            t1 = (1 - self.label) * (2 / self.Q) * self.ew * self.ew
            t2 = self.label * 2 * self.Q * tf.exp((-2.77) / self.Q * self.ew)
            loss = tf.add(t1, t2)
            self.loss = tf.reduce_mean(loss)
        tf.summary.scalar("loss", self.loss)

    def _create_train_op_graph(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _compute_acc_graph(self):
        with tf.name_scope("accuracy"):
            # calculate correct
            #预测label和真实label相同则预测正确
            temp=tf.subtract(self.predict_label,self.label)
            zeroNum=tf.cast(tf.count_nonzero(temp),dtype=float)
            sum=tf.cast(tf.size(self.label),dtype=float)
            self.accuracy=(sum-zeroNum)/sum
            tf.summary.scalar("accuracy", self.accuracy)
