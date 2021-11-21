from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten

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