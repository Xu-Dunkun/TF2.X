import tensorflow as tf
import os
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

convert_flag = 0

if convert_flag == 0:
    model_save_path = 'model_save'
elif convert_flag == 1:
    model_save_path = './model_h5/model.h5'

def convert_to_pb(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    model.summary()

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda Input: model(Input))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

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
                      name="model.pb",
                      as_text=False)