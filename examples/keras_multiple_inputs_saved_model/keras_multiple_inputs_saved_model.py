import tensorflow as tf;
from tensorflow.python import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import Model
import numpy as np

input = Input((5))
x = Flatten()(input)
x = Dense(3,'relu')(x)
dense = Model(input, x) 

input1 = Input((5), name='test_in1')
input2 = Input((5), name='test_in2') 

dense1 = dense(input1)
dense2 = dense(input2) 

merge_layer = Concatenate()([dense1, dense2]) 
dense_layer = Dense(1, activation="sigmoid", name='test_out')(merge_layer) 

model = Model(inputs=[input1, input2], outputs=dense_layer)

v1 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
v2 = np.array([[0.6, 0.7, 0.8, 0.9, 0.1]])
print(v1.shape)
print(v2.shape)

x1, x2 = np.random.randn(100, 5), np.random.randn(100, 5)
print(x1.shape)
print(x2.shape)
y = np.random.randn(100, 1)

outputs = np.array([1.0]);
model.compile(optimizer ='adam',loss='binary_crossentropy', metrics = ['accuracy'])
model.fit([v1, v2], outputs, epochs=1, batch_size=1)
model.save('examples/keras_multiple_inputs_saved_model', save_format='tf')
