import tensorflow as tf


class LinearRegresstion(tf.Module):
    def __init__(self, name=None):
        super(LinearRegresstion, self).__init__(name=name)
        self.w = tf.Variable(tf.random.uniform([1], -1.0, 1.0), name='w')
        self.b = tf.Variable(tf.zeros([1]), name='b')
        self.optimizer = tf.keras.optimizers.SGD(0.5)

    @tf.function
    def __call__(self, x):
        y_hat = self.w * x + self.b
        return y_hat

    @tf.function
    def get_w(self):
        return {'output': self.w}

    @tf.function
    def get_b(self):
        return {'output': self.b}

    @tf.function
    def train(self, x, y):
        with tf.GradientTape() as tape:
            y_hat = self(x)
            loss = tf.reduce_mean(tf.square(y_hat - y))
        grads = tape.gradient(loss, self.trainable_variables)
        _ = self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {'loss': loss}


model = LinearRegresstion()

# Get concrete functions to generate signatures
x = tf.TensorSpec([None], tf.float32, name='x')
y = tf.TensorSpec([None], tf.float32, name='y')

train = model.train.get_concrete_function(x, y)
w = model.get_w.get_concrete_function()
b = model.get_b.get_concrete_function()

signatures = {'train': train, 'w': w, 'b': b}

directory = 'examples/regression_savedmodel'
tf.saved_model.save(model, directory, signatures=signatures)