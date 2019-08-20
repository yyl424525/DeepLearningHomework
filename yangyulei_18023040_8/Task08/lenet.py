import tensorflow as tf
from tensorflow.contrib.layers import flatten

class Lenet:

    def __init__(self,learning_rate,sigma):
        self.learning_rate=learning_rate
        self.sigma=sigma
        #在创建的时候运行画图
        self._build_graph()

    #涉及网络的所有画图build graph过程,常用一个build graph封起来
    def _build_graph(self, network_name='Lenet'):
        self._setup_placeholders_graph()
        self._build_network_graph(network_name)
        self._compute_loss_graph()
        self._create_train_op_graph()
        self._compute_acc_graph()

    def _cnn_layer(self,scope_name, W_name, b_name,x,filter_shape,conv_strides, b_shape,padding_tag='VALID'):
        with tf.variable_scope(scope_name) as scope:
            conv_weights = tf.get_variable(name=W_name,shape=filter_shape,initializer=tf.truncated_normal_initializer(stddev=self.sigma))
            conv_biases = tf.get_variable(name=b_name,shape=b_shape,initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(x, conv_weights, strides=conv_strides, padding=padding_tag)
            act=tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
            tf.summary.histogram(W_name, conv_weights)
            tf.summary.histogram(b_name, conv_biases)
            return act

    def _pooling_layer(self, scope_name, relu, pool_ksize,pool_strides, padding_tag='VALID'):
        with tf.variable_scope(scope_name) as scope:
            return tf.nn.max_pool(relu, ksize=pool_ksize, strides=pool_strides, padding=padding_tag)

    def _flatten(self,pool2):
        #将x拉直
        pool_shape=pool2.get_shape().as_list()
        length= pool_shape[1] * pool_shape[2] * pool_shape[3]
        return tf.reshape(pool2,[pool_shape[0],length])

    def _fully_connected_layer(self,scope_name,W_name, b_name,x,W_shape,b_shape):
        with tf.variable_scope(scope_name) as scope:
            fc_weights = tf.get_variable(W_name,W_shape,initializer=tf.truncated_normal_initializer(stddev=self.sigma))
            fc_biases = tf.get_variable(b_name,b_shape,initializer=tf.constant_initializer(0.1))

            act = tf.nn.relu(tf.matmul(x, fc_weights) + fc_biases)
            tf.summary.histogram(W_name, fc_weights)
            tf.summary.histogram(b_name, fc_biases)
            if scope_name=="layer_6_fc2":
             with tf.name_scope('fc') as v_s:
                # scale weights to [0 1], type is still float
                x_min = tf.reduce_min(fc_weights)
                x_max = tf.reduce_max(fc_weights)
                fc_W_img = (fc_weights - x_min) / (x_max - x_min)
                fc_W_img_reshape = tf.reshape(fc_W_img, [-1, W_shape[0], W_shape[1], 1])
                tf.summary.image(W_name, fc_W_img_reshape)

            return act

    #构建图
    def _setup_placeholders_graph(self):
            self.x  = tf.placeholder(tf.float32, (None, 32, 32, 1),name='input_x')
            print(self.x.shape)
            self.y= tf.placeholder(tf.int32, (None))  # 在模型中的占位

    def _build_network_graph(self, scope_name):
        with tf.variable_scope(scope_name):
            #第一个卷积层
            # Input = 32x32x1. Output = 28x28x6.
            #卷积核：[filter_height, filter_width, in_channels, out_channels]
            self.conv1_relu= self._cnn_layer('layer_1_conv','conv1_w', 'conv1_b', self.x, (5,5,1,6),[1,1,1,1],[6])

            #第一个池化层
            #Input = 28x28x6. Output = 14x14x6.
            #图像序列 x 高 x 宽 x 通道序列;步长只设定在“高”和“宽”的维度为 2。
            self.pool1 = self._pooling_layer('layer_2_pooling', self.conv1_relu, [1, 2, 2, 1], [1, 2, 2, 1])

            #第二个卷积层
            #Output = 10x10x16.

            self.conv2_relu= self._cnn_layer('layer_3_conv', 'conv2_w', 'conv2_b', self.pool1, (5, 5,6,16), [1, 1, 1, 1],[16])

            #第二个池化层
            #Input = 10x10x16. Output = 5x5x16.
            self.pool2 = self._pooling_layer('layer_4_pooling', self.conv2_relu, [1, 2, 2, 1], [1, 2, 2, 1])

            #Tensor to vector:输入维度由 Nx5x5x16 压平后变为 Nx400
            print("self.pool2.shape:",self.pool2.shape)
            self.pool2=flatten(self.pool2)

            #第一个全连接层
            #Input = 400. Output = 120.
            self.fc1_relu=self._fully_connected_layer('layer_5_fc1','fc1_w','fc1_b',self.pool2,(400,120),[120])

            #第二个全连接层
            #Input = 120. Output = 84.
            self.fc2_relu=self._fully_connected_layer('layer_6_fc2','fc2_w','fc2_b',self.fc1_relu,(120,84),[84])

            #第三个全连接层
            #Input = 84. Output = 10.
            self.fc3_relu=self._fully_connected_layer('layer_7_fc3','fc3_w','fc3_b',self.fc2_relu,(84,10),[10])

            self.digits=self.fc3_relu

            return self.digits

    def _compute_loss_graph(self):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.digits)
            self.loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar("loss", self.loss)

    def _create_train_op_graph(self):
        # self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5)
        # 获取loss对var_list中变量的梯度
        var_list = tf.trainable_variables()
        gradients = optimizer.compute_gradients(self.loss, var_list)
        # 对梯度进行截断
        capped_gradients = [(tf.clip_by_value(grad, -0.001, 0.001), var) for grad, var in gradients if grad is not None]
        # 应用截断梯度来更新参数
        self.train_op = optimizer.apply_gradients(capped_gradients)

    def _compute_acc_graph(self):
            # calculate correct
            self.prediction = tf.equal(tf.argmax(self.digits, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.prediction, tf.float32))
            tf.summary.scalar("accuracy", self.accuracy)

