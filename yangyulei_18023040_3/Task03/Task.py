import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
#数据集由（1,784）切分成两份，每份为（1,392）
def splitX(x):
    x1,x2=tf.split(x,num_or_size_splits=2,axis=1)
    return x1,x2

def X_W(X):
    W=tf.get_variable(shape=[392,10],name='weight')
    print("W_name:",W.name)
    return tf.matmul(X, W)

def compute_y(x):
    # X1 = tf.placeholder(dtype='float', shape=[None, 392])
    # X2 = tf.placeholder(dtype='float', shape=[None, 392])
    X1,X2=splitX(x)
    with tf.variable_scope("share_weight") as scope:
        out1 = X_W(X1)
        #允许变量共享
        scope.reuse_variables()
        out2 = X_W(X2)
        y = tf.nn.softmax(out1 + out2 + b)  # 预测值
        return y

mnist = read_data_sets("data/",one_hot=True)
x = tf.placeholder(dtype='float',shape =[None ,784])
b = tf.Variable(tf.zeros ([10]))


y=compute_y(x)
y_ = tf.placeholder(dtype='float',shape =[None ,10]) #真实值
#计算交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train. GradientDescentOptimizer (learning_rate =0.01).minimize(cross_entropy)

init = tf. global_variables_initializer ()
sess = tf.Session ()
sess.run(init)


step = 500
loss_list = []
for i in range(step):
    #从训练集里一次提取100张图片数据来训练
    batch_xs ,batch_ys = mnist.train.next_batch (100) #shape: (100, 784) (100, 10)
    steps=i*100
    _,loss= sess.run([train_step ,cross_entropy ],feed_dict ={x:batch_xs,y_:batch_ys })
    loss_list.append(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print('step:',steps,'[accuracy ,loss ]:', sess.run([accuracy, cross_entropy], feed_dict={x: mnist.test.images, y_: mnist.
                                         test.labels}))




# mnist 组成:
# 60000 行训练数据集(mnist.train)和 10000 行的测试数据集
# mnist.train.images 是一个形状为 [60000, 784] 的张量，将28*28的矩阵摊平成为一个1行784列的一维数组
# mnist.train.labels 是一个[60000, 10] 的数字矩阵。