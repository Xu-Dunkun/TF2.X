import numpy as np
from PIL import Image
import os

train_path = './image_data/train_img'
train_text = './image_data/train_img.txt'
x_train_savepath = './image_data/x_train.npy'
y_train_savepath = './image_data/y_train.npy'

test_path = './image_data/test_img'
test_text = './image_data/test_img.txt'
x_test_savepath = './image_data/x_test.npy'
y_test_savepath = './image_data/y_test.npy'

def generate(path, txt):
    f = open(txt, 'r')
    contents = f.readlines()
    f.close()
    x, y = [], []

    for content in contents:
        value = content.split()
        img_path = path + '/' + value[0]
        img = Image.open(img_path)
        img = np.array(img.convert('L'))
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

else:
    print('-------------Generate Datasets-----------------')
    x_train, y_train = generate(train_path, train_text)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test, y_test = generate(test_path, test_text)

    print('-------------Save Datasets-----------------')
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    x_test_save = np.reshape(x_test, (len(x_test), -1))
    np.save(x_train_savepath, x_train_save)
    np.save(y_train_savepath, y_train)
    np.save(x_test_savepath, x_test_save)
    np.save(y_test_savepath, y_test)
