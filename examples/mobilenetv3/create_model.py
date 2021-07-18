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
fname = "examples/mobilenetv3/218/2184087daecdcafa63c8af66f02f1b360632cd80221.jpeg"
buf = tf.io.read_file(fname)
img = tf.image.decode_jpeg(buf)

# resize to (224, 224, 3)
small = tf.image.resize(img[:, 92:-75, :], (224, 224), antialias=True)

# dump the content to use from Rust later
small = tf.cast(small, tf.uint8)
buf = tf.image.encode_jpeg(small)
tf.io.write_file(directory + "/sample.jpeg", buf)

# check model prediction
predict = model(small[tf.newaxis, :, :, :])
print(tf.math.argmax(predict, axis=1))
