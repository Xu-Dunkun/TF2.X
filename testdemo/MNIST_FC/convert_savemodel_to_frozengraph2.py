from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, InputLayer
from tensorflow.keras import Model, Sequential
import os
import numpy as np
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # class MnistModel(Model):
    #     def __init__(self):
    #         super(MnistModel, self).__init__()
    #         self.i = InputLayer(input_shape=(28, 28), name="input")
    #         self.f1 = Flatten(input_shape=(28, 28), name="flatten")
    #         self.d1 = Dense(128, activation='relu', name='dense')
    #         self.d2 = Dense(10, activation='softmax', name='output')
    #
    #     def call(self, x, training=None, mask=None):
    #         x = self.i(x)
    #         x = self.f1(x)
    #         x = self.d1(x)
    #         y = self.d2(x)
    #         return y
    #
    # model = MnistModel()

    model = Sequential(layers=[InputLayer(input_shape=(28, 28), name="input"),
                               Flatten(input_shape=(28, 28), name="flatten"),
                               Dense(128, activation="relu", name="dense"),
                               Dense(10, activation="softmax", name="output")], name="FCN")

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, ),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)

    model.summary()

    # Save model to H5
    model.save("./mnist_h5_model/mnist_model.h5")

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))

    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    # layers = [op.name for op in frozen_func.graph.get_operations()]
    # print("-" * 50)
    # print("Frozen model layers: ")
    # for layer in layers:
    #     print(layer)
    #
    # print("-" * 50)
    # print("Frozen model inputs: ")
    # print(frozen_func.inputs)
    # print("Frozen model outputs: ")
    # print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./frozen_model",
                      name="mnist_model.pb",
                      as_text=False)

    # Load frozen graph using TensorFlow 1.x functions
    with tf.io.gfile.GFile("./frozen_model/mnist_model.pb", "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    # Wrap frozen graph to ConcreteFunctions
    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=["x:0"],
                                    outputs=["Identity:0"],
                                    print_graph=True)

    # Note that we only have "one" input and "output" for the loaded frozen function
    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

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

        # Get predictions
        frozen_graph_predictions = frozen_func(x=tf.constant(x_predict))[0]
        # frozen_graph_predictions_priority = frozen_graph_predictions[0]
        # frozen_graph_predictions_department = frozen_graph_predictions[1]

        result = frozen_graph_predictions
        print(result)
        result = np.squeeze(result)
        print(result)
        pred = np.argmax(result)
        print(pred)
        print("置信值:{}".format(result[pred]))
        print('\n')
        tf.print(pred)

def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    print("-" * 50)
    print("Frozen model layers: ")
    layers = [op.name for op in import_graph.get_operations()]
    if print_graph == True:
        for layer in layers:
            print(layer)
    print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


if __name__ == "__main__":
    main()
