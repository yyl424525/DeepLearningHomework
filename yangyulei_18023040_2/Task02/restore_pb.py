import tensorflow as tf

with tf.Session() as sess:
  with tf.gfile.FastGFile("save_pb.pb", "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    # result, x = tf.import_graph_def(graph_def, return_elements=["add:0", "input:0"])
    sess.graph.as_default()
    tf.import_graph_def(graph_def,name='')
    sess.run(tf.global_variables_initializer())
    print(sess.run(sess.graph.get_tensor_by_name("w1:0")))
    print(sess.run(sess.graph.get_tensor_by_name("w2:0")))
    print(sess.run(sess.graph.get_tensor_by_name("w3:0")))
    print(sess.run(sess.graph.get_tensor_by_name("b:0")))

