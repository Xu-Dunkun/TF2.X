import tensorflow as tf
import os
from tensorflow.keras.layers import Conv2D, Flatten, ReLU, Dense, BatchNormalization, MaxPool2D, Dropout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

def main():

    model = tf.keras.models.load_model('./model_save')
    convert_from_quant_keras_model_to_tflite(model, x_train, './tflite_quant', '/quant_weight.tflite')

def convert_from_quant_keras_model_to_tflite(model, resp_data, dirname, filename):
    def representative_data_gen():
        for data in tf.data.Dataset.from_tensor_slices(resp_data).batch(1).take(100):
            yield [tf.dtypes.cast(data, tf.float32)]

    # Converting a tf.Keras model to a TensorFlow Lite model.
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # tflite_model_quant = converter.convert()

    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.representative_dataset = representative_data_gen
    # tflite_model_quant = converter.convert()

    #To quant weights / variables / input / output tensors
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model_quant = converter.convert()

    #converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

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

if __name__ == '__main__':
    main()


import tensorflow_model_optimization as tfmot
model = tf.keras.Sequential([...])
quantized_model = tfmot.quantization.keras.quantize_model(model)
quantized_model.compile(...)
quantized_model.fit(...)

import tensorflow_model_optimization as tfmot
quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
model = tf.keras.Sequential([quantize_annotate_layer(Conv2D()),
                             quantize_annotate_layer(BatchNormalization()),
                             quantize_annotate_layer(ReLU()),
                             quantize_annotate_layer(MaxPool2D),
                             quantize_annotate_layer(Dropout),
                             ...,
                             ...,
                             ...,
                             Dense()])
quantized_model = tfmot.quantization.keras.quantize_apply(model)
quantized_model.compile(...)
quantized_model.fit(...)

import tensorflow_model_optimization as tfmot
cluster_weights = tfmot.clustering.keras.cluster_weights
pretrained_bias = tfmot.clustering.keras.clusterable_layer
pretrained_model = model()
clustering_params = {
    'number_of_clusters': 32,
    'cluster_centroids_init': tfmot.clustering.keras.CentroidInitialization.LINEAR
}
clustered_model = cluster_weights(pretrained_model, **clustering_params)

clustered_model.fit(...)
model_for_serving = tfmot.clustering.keras.strip_clustering(clustered_model)


clustered_model = tf.keras.Sequential([
      Dense(...),
      cluster_weights(Dense(...,
                      kernel_initializer=cluster_weights,
                      bias_initializer=pretrained_bias),
                      **clustering_params),
      Dense(...)])