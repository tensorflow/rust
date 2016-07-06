import os
import tensorflow as tf

a = tf.placeholder(tf.float32, name='a')
b = tf.placeholder(tf.float32, name='b')
c = tf.mul(a, b, name='c')

definition = tf.Session().graph_def
directory = os.path.dirname(os.path.realpath(__file__))
tf.train.write_graph(definition, directory, 'multiplication.pb', as_text=False)
