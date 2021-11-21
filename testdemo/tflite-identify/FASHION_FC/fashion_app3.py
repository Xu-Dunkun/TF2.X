from PIL import Image
import numpy as np
import tensorflow as tf
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    # Load frozen graph using TensorFlow 1.x functions
    with tf.io.gfile.GFile("./frozen_model/model.pb", "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    # Wrap frozen graph to ConcreteFunctions
    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=["Input:0"],
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
        img_path = 'pred/'
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
        x_predict = img_arr.reshape([1, 28, 28, 1])
        print("x_predict:", x_predict.shape)

        t1 = time.time()
        # Get predictions
        frozen_graph_predictions = frozen_func(Input=tf.constant(x_predict, tf.float32))[0]
        # frozen_graph_predictions_priority = frozen_graph_predictions[0]
        # frozen_graph_predictions_department = frozen_graph_predictions[1]

        result = frozen_graph_predictions
        t2 = time.time()
        print('inference time: ' + str((t2-t1)*1000) + " ms")
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

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

if __name__  == '__main__':
    main()
