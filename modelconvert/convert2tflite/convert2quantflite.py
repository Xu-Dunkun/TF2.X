import tensorflow as tf
import os
import pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    global converter
    load_flag = 0
    quant_flag = 0
    resp_data = 1

    if load_flag == 0:
        model = tf.keras.models.load_model('./model_save')
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    elif load_flag == 1:
        model = tf.keras.models.load_model('./model_h5/model.h5')
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    elif load_flag == 2:
        model = load_frozen_graph('./frozen_graph/model.pb')
        converter = tf.lite.TFLiteConverter.from_concrete_functions(model)

    if quant_flag == 0:
        convert_quant_weight_model_to_tflite(converter, './tflite_post_quant', '/quant_weight.tflite')
        print("---  convert quant_weight_model success  ---")
    elif quant_flag == 1:
        convert_quant_full_integer_model_to_tflite(converter, resp_data, './tflite_post_quant', '/quant_full_integer.tflite')
        print("---  convert quant_full_integer_model success  ---")
    elif quant_flag == 2:
        convert_quant_full_integer_model_to_tflite(converter, resp_data, './tflite_post_quant', '/quant_only_integer.tflite')
        print("---  convert quant_only_integer_model success  ---")


def convert_quant_weight_model_to_tflite(converter, dirname, filename):

    # Converting a tf.Keras model to a TensorFlow Lite model.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # default quant int8
    #converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    save_quant_tflite(tflite_model, dirname, filename)

def convert_quant_full_integer_model_to_tflite(converter, images, dirname, filename):
    def representative_dataset():
        for data in tf.data.Dataset.from_tensor_slices((images)).batch(1).take(100):
            yield [tf.dtypes.cast(data, tf.float32)]

    #To quant weights / variables / input / output tensors
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    # default quant int8
    # converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    save_quant_tflite(tflite_model, dirname, filename)

def convert_quant_integer_only_model_to_tflite(model, images, dirname, filename):
    def representative_dataset():
        for data in tf.data.Dataset.from_tensor_slices((images)).batch(1).take(100):
            yield [tf.dtypes.cast(data, tf.float32)]

    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    # default quant int8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.target_spec.supported_types = [tf.float16]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8

    tflite_model = converter.convert()
    save_quant_tflite(tflite_model, dirname, filename)

def save_quant_tflite(tflite_model, dirname, filename, ):
    tflite_models_dir = pathlib.Path(dirname)
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir + filename
    tflite_model_file.write_bytes(tflite_model)

def load_frozen_graph(frozen_graph_path):
    with tf.io.gfile.GFile(frozen_graph_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    # Wrap frozen graph to ConcreteFunctions
    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=["Input:0"],
                                    outputs=["Identity:0"],
                                    print_graph=True)
    return frozen_func

def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=True):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

if __name__ == '__main__':
    main()
