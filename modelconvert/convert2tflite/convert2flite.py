import tensorflow as tf
import os
import pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():

    load_flag = 0
    load_path = './model_save'   #'./model_h5/model.h5' './frozen_graph/model.pb'
    dirname = './tflite'
    filename = 'save_model.tflite'  #'h5 .tflite' 'frozen .tflite'

    convert_unquant_model_to_tflite(load_flag, load_path, dirname, filename)
    print("---  convert {} success  ---".format(filename))

def convert_unquant_model_to_tflite(load_flag, load_path, dirname, filename):
    global converter
    if load_flag == 0:
        model = tf.keras.models.load_model(load_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    elif load_flag == 1:
        model = tf.keras.models.load_model(load_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    elif load_flag == 2:
        model = load_frozen_graph(load_path)
        converter = tf.lite.TFLiteConverter.from_concrete_functions(model)

    tflite_model = converter.convert()
    save_tflite_model(tflite_model, dirname, filename)

def save_tflite_model(tflite_model, dirname, filename):
    tflite_models_dir = pathlib.Path(dirname)
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir + '/' + filename
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
