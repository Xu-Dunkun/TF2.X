import tensorflow as tf
import numpy as np
from PIL import Image
import os

train_path = './mnist_image_data/mnist_train_jpg_60000/'
train_text = './mnist_image_data/mnist_train_jpg_60000.txt'
x_train_savepath = './mnist_image_data/mnist_x_train.npy'
y_train_savepath = './mnist_image_data/mnist_y_train.npy'

test_path = './mnist_image_data/mnist_test_jpg_10000/'
test_text = './mnist_image_data/mnist_test_jpg_10000.txt'
x_test_savepath = './mnist_image_data/mnist_x_test.npy'
y_test_savepath = './mnist_image_data/mnist_y_test.npy'

def generateds(path, txt):
    f = open(txt, 'r')
    contents = f.readlines()
    f.close()
    x, y = [], []

    for content in contents:
        value = content.split()
        img_path = path + value[0]
        img = Image.open(img_path)
        img = np.array(img.convert('L'))
        img = img / 255.0
        # print("img:" + img)
        x.append(img)
        y.append(value[1])
        print('loading : ' + content)

    x = np.array(x)
    y = np.array(y)
    y = y.astype(np.int64)

    return x, y

if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(
        x_test_savepath) and os.path.exists(y_test_savepath):
    print('-------------Load Datasets-----------------')
    x_train_save = np.load(x_train_savepath)
    y_train_save = np.load(y_train_savepath)
    x_test_save = np.load(x_test_savepath)
    y_test_save = np.load(y_test_savepath)
    x_train = np.reshape(x_train_save, (len(x_train_save), 28, 28))
    x_test = np.reshape(x_test_save, (len(x_test_save), 28, 28))
else:
    print('-------------Generate Datasets-----------------')
    x_train, y_train = generateds(train_path, train_text)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test, y_test = generateds(test_path, test_text)

    print('-------------Save Datasets-----------------')
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    x_test_save = np.reshape(x_test, (len(x_test), -1))
    np.save(x_train_savepath, x_train_save)
    np.save(y_train_savepath, y_train)
    np.save(x_test_savepath, x_test_save)
    np.save(y_test_savepath, y_test)