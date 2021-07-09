# TODO: Stop using v1 compatibility
import tensorflow.compat.v1 as tf


tf.disable_eager_execution()
x = tf.placeholder(tf.int32, name = 'x')
y = tf.placeholder(tf.int32, name = 'y')
z = tf.add(x, y, name = 'z')

tf.variables_initializer(tf.global_variables(), name = 'init')

definition = tf.Session().graph_def
directory = 'examples/addition'
tf.train.write_graph(definition, directory, 'model.pb', as_text=False)
