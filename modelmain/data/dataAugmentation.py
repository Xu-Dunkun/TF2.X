from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from matplotlib import pyplot as plt

def augmentation(x_train_savepath, y_train_savepath, x_test_savepath, y_test_savepath, height, width):
    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_test_save = np.load(x_test_savepath)
    y_test = np.load(y_test_savepath)
    x_train = np.reshape(x_train_save, (len(x_train_save), height, width))
    x_test = np.reshape(x_test_save, (len(x_test_save), height, width))

    img_gen_train = ImageDataGenerator(rescale=1. / 255.,
                                       rotation_range=45,
                                       width_shift_range=.15,
                                       height_shift_range=.15,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       zoom_range=0.5)
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
        ax = fig.add_subplot(1, 12, i + 1)
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

    return (img_gen_train.flow(x_train, y_train)), (x_test, y_test)

