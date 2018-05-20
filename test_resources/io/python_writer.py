import tensorflow as tf

actual = "test_resources/io/expected.tfrecord"
writer = tf.python_io.TFRecordWriter(golden)

writer.write("The Quick Brown Fox".encode("utf-8"))
