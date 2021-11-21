from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model_save_path = './checkpoint/mnist.ckpt'

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

model.load_weights(model_save_path)

preNum = int(input("input the number of test pictures:"))

for i in range(preNum):
    img_path = 'mnist_image_data/mnist_pre/'
    img_path = img_path + input("the path of test picture:")
    img = Image.open(img_path)
    img = img.resize((28, 28), Image.ANTIALIAS)
    img_arr = np.array(img.convert('L'))

    for i in range(28):
        for j in range(28):
            if img_arr[i][j] < 200:
                img_arr[i][j] = 255
            else:
                img_arr[i][j] = 0

    img_arr = img_arr / 255.0
    x_predict = img_arr[tf.newaxis, ...]
    print("x_predict:", x_predict.shape)
    result = model.predict(x_predict)
    result = np.squeeze(result)
    print(result)
    pred = np.argmax(result)
    print(pred)
    print("置信值:{}".format(result[pred]))
    print('\n')
    tf.print(pred)

