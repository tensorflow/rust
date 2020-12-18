import tensorflow as tf;
import tensorboard;
from tensorflow.python import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

import json;
import numpy as np;


tensorboard_callback = keras.callbacks.TensorBoard(log_dir="/tmp/logs")

classifier = Sequential()
classifier.add(Dense(5, activation='relu', kernel_initializer='random_normal', name="test_in", input_dim=5))
classifier.add(Dense(5, activation='relu'))
classifier.add(Dense(1, activation='sigmoid', name="test_out"))



classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics = ['accuracy'])


classifier.fit([[0.1, 0.2, 0.3, 0.4, 0.5]], [[1]], batch_size=1, epochs=1, callbacks=[tensorboard_callback]);
result = classifier.predict([[0.1, 0.2, 0.3, 0.4, 0.5]])

print(result);
classifier.summary();

for layer in classifier.layers:
    print(layer.name)

classifier.save('examples/keras_signle_input_saved_model', save_format='tf')
