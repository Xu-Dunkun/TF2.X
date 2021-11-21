import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
import os
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(threshold=np.inf)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

# image_gen_train = ImageDataGenerator(rescale=1. / 1.,
#                                      rotation_range=45,
#                                      width_shift_range=.15,
#                                      height_shift_range=.15,
#                                      horizontal_flip=True,
#                                      zoom_range=0.5)

# image_gen_train.fit(x_train)

class MnistModel(Model):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.f1 = Flatten()
        self.d1 = Dense(128, activation='relu')
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
                                                 save_weights_only=False,
                                                 save_best_only=True)

# model_history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=32), epochs=5,
#           validation_data=(x_test, y_test), validation_freq=1, callbacks=[cp_callback])

model_history = model.fit(x_train, y_train, batch_size=32, epochs=5,
                          validation_data=(x_test, y_test), validation_freq=1, callbacks=[cp_callback])

model.summary()

tf.keras.models.save_model(model, "model_save")
print('模型已保存')

file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

acc = model_history.history['sparse_categorical_accuracy']
val_acc = model_history.history['val_sparse_categorical_accuracy']
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()


