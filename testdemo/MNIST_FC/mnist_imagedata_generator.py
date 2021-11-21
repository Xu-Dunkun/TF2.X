import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

train_path = './mnist_image_data/mnist_train_jpg_60000'
train_text = './mnist_image_data/mnist_train_jpg_60000.txt'
test_path = './mnist_image_data/mnist_test_jpg_10000'
test_text = './mnist_image_data/mnist_test_jpg_10000.txt'

def generateds(path, txt):
    f = open(txt, 'r')
    contents = f.readlines()
    f.close()
    x, y = [], []

    for content in contents:
        value = content.split()
        img_path = path + '/' + value[0]
        img = Image.open(img_path)
        img = np.array(img.convert('L'))
        img = img / 255.0
        x.append(img)
        y.append(value[1])
        print('loading : ' + content)

    x = np.array(x)
    y = np.array(y)
    y = y.astype(np.int64)
    return x, y

x_train, y_train = generateds(train_path, train_text)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test, y_test = generateds(test_path, test_text)

img_gen_train = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.5
)

img_gen_train.fit(x_train)
print('xtrain', x_train.shape)
x_train_subset1 = np.squeeze(x_train[:12])
print("xtrain_subset1", x_train_subset1.shape)

print("xtrain", x_train.shape)
x_train_subset2 = x_train[:12]
print("xtrain_subset2", x_train_subset2.shape)

fig = plt.figure(figsize=(20, 2))
plt.set_cmap('gray')
for i in range(0, len(x_train_subset1)):
    ax = fig.add_subplot(1, 12, i+1)
    ax.imshow(x_train_subset1[i])
fig.suptitle('Subset of Original Training Images', fontsize=20)
plt.show()

fig = plt.figure(figsize=(20, 2))
for x_batch in img_gen_train.flow(x_train_subset2, batch_size=12, shuffle=False):
    for i in range(0, 12):
        ax = fig.add_subplot(1, 12, i + 1)
        ax.imshow(np.squeeze(x_batch[i]))
    fig.suptitle('Augmented Images', fontsize=20)
    plt.show()
    break

