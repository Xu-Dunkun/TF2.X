import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

def main():
    with tf.io.gfile.GFile("./frozen_model/model.pb", "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    # Wrap frozen graph to ConcreteFunctions
    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=["Input:0"],
                                    outputs=["Identity:0"],
                                    print_graph=True)

    convert_from_quant_keras_model_to_tflite(frozen_func, x_train, './tflite_quant', '/weight_variables_activation.tflite')

def convert_from_quant_keras_model_to_tflite(model, resp_data, dirname, filename):
    def representative_data_gen():
        for data in tf.data.Dataset.from_tensor_slices(resp_data).batch(1).take(100):
            # Model has only one input so each data point has one element.
            yield [tf.dtypes.cast(data, tf.float32)]

    # Converting a tf.Keras model to a TensorFlow Lite model.
    converter = tf.lite.TFLiteConverter.from_concrete_functions(model)

    # To quant weights: ERROR (occured when using vela)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model_quant = converter.convert()

    # # To quant weights and variables: ERROR (occured when using vela)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.representative_dataset = representative_data_gen
    # tflite_model_quant = converter.convert()

    # # To quant weights / variables / input / output tensors
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.representative_dataset = representative_data_gen
    # ### Ensure that if any ops can't be quantized, the converter throws an error
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # #converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    # #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    #
    # ### Set the input and output tensors to uint8 (APIs added in r2.3)
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8
    # tflite_model_quant = converter.convert()

    if os.path.exists(dirname):
        open(dirname + filename, "wb").write(tflite_model_quant)
    else:
        mkdir(dirname)

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print("---  There is this folder!  ---")

def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

if __name__ == '__main__':
    main()

