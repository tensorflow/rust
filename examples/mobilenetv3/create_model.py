import tensorflow as tf

# default input shape 224x224x3
model = tf.keras.applications.MobileNetV3Small(weights="imagenet")

# save the model
directory = "examples/mobilenetv3"
model.save(directory, save_format="tf")

######################################################
# Check the prediction results for the sample image. #
######################################################
# load sample image
fname = "examples/mobilenetv3/sample_image/cat.jpg"
buf = tf.io.read_file(fname)
img = tf.image.decode_jpeg(buf)

# clip to the square and resize to (224, 224)
small = tf.image.resize(img[:, 50:-150], (224, 224), antialias=True)

# dump the content to use from Rust later
small = tf.cast(small, tf.uint8)
buf = tf.image.encode_png(small)
tf.io.write_file(directory + "/sample.png", buf)

# check model prediction
predict = model(small[tf.newaxis, :, :, :])
print(tf.math.argmax(predict, axis=1))
