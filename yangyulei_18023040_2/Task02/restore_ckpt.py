import tensorflow as tf

w1= tf.Variable(tf.random_uniform([1]),name='w1')
w2= tf.Variable(tf.random_uniform([1]),name='w2')
w3= tf.Variable(tf.random_uniform([1]),name='w3')
b= tf.Variable(tf.random_uniform([1]),name='b')

# 这里不需要初始化步骤 init= tf.initialize_all_variables()

saver = tf.train.Saver()
with tf.Session()as sess:
  saver.restore(sess,'save_ckpt/save.ckpt')
  print('w1=', sess.run(w1), ' w2=', sess.run(w2), ' w3=', sess.run(w3), ' b3=', sess.run(b))

