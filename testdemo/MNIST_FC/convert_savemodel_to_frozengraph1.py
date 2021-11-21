from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, InputLayer
from tensorflow.keras import Model, Sequential
import os
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

    # Save model to SavedModel format
    tf.saved_model.save(model, "./mnist_model_save")

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda Input: model(Input))

    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./frozen_model",
                      name="mnist_model.pb",
                      as_text=False)

# def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
#     def _imports_graph_def():
#         tf.compat.v1.import_graph_def(graph_def, name="")
#
#     wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
#     import_graph = wrapped_import.graph
#
#     print("-" * 50)
#     print("Frozen model layers: ")
#     layers = [op.name for op in import_graph.get_operations()]
#     if print_graph == True:
#         for layer in layers:
#             print(layer)
#     print("-" * 50)
#
#     return wrapped_import.prune(
#         tf.nest.map_structure(import_graph.as_graph_element, inputs),
#         tf.nest.map_structure(import_graph.as_graph_element, outputs))

if __name__ == "__main__":

    main()