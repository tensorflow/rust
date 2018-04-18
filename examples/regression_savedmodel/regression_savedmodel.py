import tensorflow as tf
from tensorflow.python.saved_model.builder import SavedModelBuilder
from tensorflow.python.saved_model.signature_def_utils import build_signature_def
from tensorflow.python.saved_model.signature_constants import REGRESS_METHOD_NAME
from tensorflow.python.saved_model.tag_constants import TRAINING, SERVING
from tensorflow.python.saved_model.utils import build_tensor_info

x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')

w = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='w')
b = tf.Variable(tf.zeros([1]), name='b')
y_hat = tf.add(w * x, b, name="y_hat")

loss = tf.reduce_mean(tf.square(y_hat - y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss, name='train')

init = tf.variables_initializer(tf.global_variables(), name='init')

directory = 'examples/regression_savedmodel'
builder = SavedModelBuilder(directory)

with tf.Session(graph=tf.get_default_graph()) as sess:
    sess.run(init)

    signature_inputs = {
        "x": build_tensor_info(x),
        "y": build_tensor_info(y)
    }
    signature_outputs = {
        "out": build_tensor_info(y_hat)
    }
    signature_def = build_signature_def(
        signature_inputs, signature_outputs,
        REGRESS_METHOD_NAME)
    builder.add_meta_graph_and_variables(
        sess, [TRAINING, SERVING],
        signature_def_map={
            REGRESS_METHOD_NAME: signature_def
        },
        assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))
    builder.save(as_text=False)
