import tensorflow_model_optimization as tfmot
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

quantize_model = tfmot.quantization.keras.quantize_model

model = Sequential(layers=[Conv2D(filters=6, kernel_size=(5, 5), strides=1, padding='same', name='conv1'),
                           BatchNormalization(name='bn1'),
                           Activation(activation='relu', name='activation1'),
                           MaxPool2D(pool_size=(2, 2), strides=2, padding='same', name='maxpool1'),
                           Dropout(0.2, name='drop1'),
                           Flatten(name='flatten'),
                           Dense(128, activation='relu', name="f1"),
                           Dropout(0.2),
                           Dense(10, activation='softmax', name='f2')])

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

q_aware_model.summary()

