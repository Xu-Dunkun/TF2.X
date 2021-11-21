import tensorflow as tf
import glob as gb
from PIL import Image
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def generateds(train_img_path):
    x = []
    # Returns a list of all folders with participant numbers
    img_path = gb.glob(train_img_path)
    for path in img_path:
        img = Image.open(path)
        img = img.resize((128, 128), Image.BILINEAR)
        img = np.array(img)
        img = img / 255.0
        x.append(img)

    x = np.array(x)
    return x

def main():
    x_train = generateds("./train_img/*.jpg")
    x_train = x_train.reshape(x_train.shape[0], 3, 128, 128)
    convert_from_quant_keras_model_to_tflite('./shuffle2_sm_sim', x_train, './tflite_quant', '/quant_shuffle2_sm_sim.tflite')

def convert_from_quant_keras_model_to_tflite(modelfile, resp_data, dirname, filename):
    def representative_data_gen():
        for data in tf.data.Dataset.from_tensor_slices(resp_data).batch(1).take(100):
            # Model has only one input so each data point has one element.
            yield [tf.dtypes.cast(data, tf.float32)]

    # Converting a tf.Keras model to a TensorFlow Lite model.
    converter = tf.lite.TFLiteConverter.from_saved_model(modelfile)

    # set the optimization model
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # pass the representative dataset to the converter
    converter.representative_dataset = representative_data_gen

    # converter.representative_dataset = tf.lite.RepresentativeDataset(generator)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    # ### Set the input and output tensors to uint8 (APIs added in r2.3)
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8
    tflite_model_quant = converter.convert()

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

if __name__ == '__main__':
    main()