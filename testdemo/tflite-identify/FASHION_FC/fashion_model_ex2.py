import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(threshold=np.inf)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


model = Sequential([Conv2D(filters=6, kernel_size=(5, 5), strides=1, padding='same', name='conv1'),
                    BatchNormalization(name='bn1'),
                    Activation(activation='relu', name='activation1'),
                    MaxPool2D(pool_size=(2, 2), strides=2, padding='same', name='maxpool1'),
                    Dropout(0.2, name='drop1'),
                    Flatten(name='flatten'),
                    Dense(128, activation='relu', name="f1"),
                    Dropout(0.2),
                    Dense(10, activation='softmax', name='f2')])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = './checkpoint/fashion.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

model_history = model.fit(x_train, y_train, batch_size=32, epochs=5,
                          validation_data=(x_test, y_test), validation_freq=1, callbacks=[cp_callback])

model.summary()

tf.keras.models.save_model(model, './h5_model/model.h5')
# model.save('./h5_model/model.h5')
print('模型已保存')