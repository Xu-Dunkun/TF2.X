import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    model = tf.keras.models.load_model('model_save')
    convert_from_unquant_keras_model_to_tflite(model, './tflite_unquant', '/converted_from_unquant_h5.tflite')

def convert_from_unquant_keras_model_to_tflite(model, dirname, filename):
    # Converting a tf.Keras model to a TensorFlow Lite model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    if os.path.exists(dirname):
        open(dirname+filename, "wb").write(tflite_model)
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


