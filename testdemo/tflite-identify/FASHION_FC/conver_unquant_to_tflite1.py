import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    model = tf.keras.models.load_model('model_save')
    convert_from_unquant_keras_model_to_tflite(model, './tflite_unquant', '/converted_from_unquant_save_model.tflite')

def convert_from_unquant_keras_model_to_tflite(model, dirname, filename):
    '''
    OK
    :param model:
    :return:
    '''
    # Converting a tf.Keras model to a TensorFlow Lite model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    if os.path.exists(dirname):
        open(dirname+filename, "wb").write(tflite_model)
    else:
        mkdir(dirname)

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")

if __name__ == '__main__':
    main()


