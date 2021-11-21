import tensorflow as tf
import os
from PIL import Image
import numpy as np

def main():

    height = 228
    width = 228
    img_path = '/data/img_pre'
    frozen_model_path = "./frozen_model/model.pb"

    with tf.io.gfile.GFile(frozen_model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    # Wrap frozen graph to ConcreteFunction
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

    for img_name in os.listdir(img_path):
        img_path = img_path + '/' + img_name
        img = Image.open(img_path)
        img = img.resize((height, width), Image.ANTIALIAS)
        img_arr = np.array(img.convert('L'))
        img_arr = img_arr / 255.0
        x_predict = img_arr[tf.newaxis, ...]
        print("x_predict:", x_predict.shape)

        # Get predictions
        frozen_graph_predictions = frozen_func(x=tf.constant(x_predict))[0]
        # frozen_graph_predictions_priority = frozen_graph_predictions[0]
        # frozen_graph_predictions_department = frozen_graph_predictions[1]

        result = frozen_graph_predictions
        result = np.squeeze(result)
        pred = np.argmax(result)
        print("结果：{}, 预测：{}, 置信值:{}".format(result, pred, result[pred]))
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