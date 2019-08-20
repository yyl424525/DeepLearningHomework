#基础作业
import math
import  numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import graph_util

a=18 #学号前两位
b=40 #学号后两位
step=((2*math.pi-b)/a+b/a)/2000 #计算步长
x=np.arange(2000.)
y=np.arange(2000.)
for i in range(0,2000):
     x[i]=-b/a+i*step
     y[i]=math.cos(a*x[i]+b)
     print("第",i,"个采样点：(",x[i],",",y[i],")")

# ----------训练------------------
# 生成参数
w1= tf.Variable(tf.random_uniform([1]),name='w1')
w2= tf.Variable(tf.random_uniform([1]),name='w2')
w3= tf.Variable(tf.random_uniform([1]),name='w3')
b= tf.Variable(tf.random_uniform([1]),name='b')
print(w1)
y_=w1*x*x*x+w2*x*x+w3*x+b

# 梯度下降法优化参数(学习率为lr)
lr = 0.01
# 定义loss:均方差
loss = tf.reduce_mean(tf.square(y - y_),name="loss")

optimizer = tf.train.GradientDescentOptimizer(lr)
# 训练的过程就是最小化这个误差值
train = optimizer.minimize(loss)

#初始化全局变量
# init= tf.initialize_all_variables() # tf 马上就要废弃这种写法
# 替换成下面的写法:
init = tf.global_variables_initializer()

# 通过session执行上述操作
sess = tf.Session()
sess.run(init)

#初始化的值
print('w1=', sess.run(w1), ' w2=', sess.run(w2), ' w3=', sess.run(w3), ' b3=', sess.run(b), ' loss=', sess.run(loss))

i=0
#设置loss大于0.05时继续训练
while (sess.run(loss) > 0.05):
    #每次迭代都要最小化Loss函数
    sess.run(train)
    # 输出训练好的w1 w2 w3 b
    print('第',i,'次迭代时参数和loss的值为：','w1=', sess.run(w1), ' w2=', sess.run(w2), ' w3=', sess.run(w3), ' b3=', sess.run(b), ' loss=',sess.run(loss))
    i=i+1

plt.scatter(x, y, c='r')
y_=sess.run(w1)*x*x*x+sess.run(w2)*x*x+sess.run(w3)*x+sess.run(b)
# 拟合曲线
plt.figure()
#构造散点图
plot1 = plt.scatter(x, y)
#画出模拟的曲线
plot2 = plt.plot(x, y_, 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

saver = tf.train.Saver()

#ckpt模式保存
save_path = saver.save(sess,'save_ckpt/save.ckpt')

#PB模式保存
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,["w1","w2","w3","b"])
with tf.gfile.FastGFile('save_pb.pb', mode='wb') as f:
    f.write(constant_graph.SerializeToString())

#清除默认图的堆栈，并设置全局图为默认图
#解决多次运行之后的错误：NotFoundError (see above for traceback): Key Variable_1 not found in checkpoint
tf.reset_default_graph()
