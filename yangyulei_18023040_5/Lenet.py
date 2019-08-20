import tensorflow as tf
import tensorflow.contrib.slim as slim


# 通过TensorFlow-Slim来定义LeNet-5的网络结构。
def lenet(images):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        inputs = tf.reshape(images, [-1, 28, 28, 1])
        net = slim.conv2d(inputs, 32, [5, 5], padding='SAME', scope='layer1-conv')
        net = slim.max_pool2d(net, 2, stride=2, scope='layer2-max-pool')
        net = slim.conv2d(net, 64, [5, 5], padding='SAME', scope='layer3-conv')
        net = slim.max_pool2d(net, 2, stride=2, scope='layer4-max-pool')
        net = slim.flatten(net, scope='flatten')
        net = slim.fully_connected(net, 500, scope='layer5')
        net = slim.fully_connected(net, 10, scope='output')
    return net


