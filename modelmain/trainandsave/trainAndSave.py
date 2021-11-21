import tensorflow as tf
import os
from matplotlib import pyplot as plt
from data.dataAugmentation import augmentation as aug
from modelstructure.model import MnistModel
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x_train_savepath = '/data/image_data/'
y_train_savepath = '/data/image_data/'
x_test_savepath = '/data/image_data/'
y_test_savepath = '/data/image_data/'

bt_size = 64
train_epochs = 10
resize_height = 228
resize_width = 228

save_flag = 0

(x_train, y_train), (x_test, y_test) = aug(x_train_savepath, y_train_savepath, x_test_savepath, y_test_savepath, resize_height, resize_width)
x_train, x_test = x_train/255., x_test/255.

model = MnistModel()

checkpoint_save_path = './checkpoint/model.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=False,
                                                 save_best_only=True)

model_history = model.fit(x_train, y_train, batch_size=bt_size, epochs=train_epochs,
                          validation_data=(x_test, y_test), validation_freq=1, callbacks=[cp_callback])

model.summary()

if(save_flag == 0):
    tf.keras.models.save_model(model, "model_save")
    print('model_save 模型已保存')
else:
    model.save("model_h5/model.h5")
    print('model_h5 模型已保存')

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
