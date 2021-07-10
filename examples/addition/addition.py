import tensorflow as tf

# check tensorflow version is 2.x 
tf_major_version = tf.__version__.split('.')[0]
assert tf_major_version == '2'

@tf.function
def add(x, y):
    return tf.add(x, y)

x = tf.TensorSpec((), dtype=tf.dtypes.int32, name='x')
y = tf.TensorSpec((), dtype=tf.dtypes.int32, name='y')

concrete_function = add.get_concrete_function(x, y)
directory = 'examples/addition'
tf.io.write_graph(concrete_function.graph, directory, 'model.pb', as_text=False)

# check inputs/outputs node names to refer from Rust later on
print(f'input nodes  : {concrete_function.inputs}')
print(f'output nodes : {concrete_function.outputs}')