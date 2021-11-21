import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
import os

np.set_printoptions(threshold=np.inf)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

image_gen_train = ImageDataGenerator(
    rescale=1. / 1.,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.5
)

image_gen_train.fit(x_train)

class MnistModel(Model):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.f1 = Flatten()
        self.d1 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())
        self.d2 = Dense(10, activation='softmax')

    def call(self, x, training=None, mask=None):
        x = self.f1(x)
        x = self.d1(x)
        y = self.d2(x)

        return y

model = MnistModel()

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = './checkpoint/mnist.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 monitor='loss',
                                                 save_best_only=True)

history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=32), epochs=5,
          validation_data=(x_test, y_test), validation_freq=1, callbacks=[cp_callback])

model.summary()

print(model.trainable_variables)
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()