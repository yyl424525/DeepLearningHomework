#进阶作业
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

x = tf.Variable(1,dtype = tf.float32)  # 定义一个可以优化的x值
#定义y值：y = 1 - sin(x)/x
y = tf.subtract(1.,tf.divide(tf.sin(x),x))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(y)

init = tf.global_variables_initializer()
# 生成会话，训练1000轮
i=0
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_step)
        x_val = sess.run(x)
        y_val = sess.run(y)
        print("epoch",i,":",sess.run(x)," ",sess.run(y))
    i=i+1

plt.figure(figsize=(6, 4))  # 设置图片大小
x_data=np.linspace(-5,5,10000)  #这个表示在0到5之间生成10000个x值
#错误：math.sin(x_data)
plt.plot(x_data,1-(np.sin(x_data)/x_data),color='red',)
plt.scatter(x_val, y_val, marker = 'x', s = 40 ,label = 'min')
plt.legend() # 显示图例
plt.show()