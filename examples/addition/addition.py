import tensorflow as tf


x = tf.placeholder(tf.int32, name = 'x')
y = tf.placeholder(tf.int32, name = 'y')
z = tf.add(x, y, name = 'z')

tf.variables_initializer(tf.global_variables(), name = 'init')

definition = tf.Session().graph_def
directory = 'examples/addition'
tf.train.write_graph(definition, directory, 'model.pb', as_text=False)
