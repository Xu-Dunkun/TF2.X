import tensorflow as tf
import os
from PIL import Image
import numpy as np
from modelstructure.model import MnistModel
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    height = 228
    width = 228
    load_flag = 0
    img_path = '/data/image_data/img_pre'
    # if load_flag == 0:
    #     model_save_path = './checkpoint/model.ckpt'
    # elif load_flag == 1:
    #     model_save_path = 'model_save'
    # elif load_flag == 2:
    #     model_save_path = './model_h5/model.h5'
    if load_flag == 0:
        model = MnistModel()
        model.load_weights('trainandsave/checkpoint/model.ckpt')
    elif load_flag == 1:
        model = tf.keras.models.load_model('trainandsave/model_save')
    elif load_flag == 2:
        model = tf.keras.models.load_model('trainandsave/model_h5/model.h5')

    for img_name in os.listdir(img_path):
        img_path = img_path + '/' + img_name
        img = Image.open(img_path)
        img = img.resize((height, width), Image.ANTIALIAS)
        img_arr = np.array(img.convert('L'))
        img_arr = img_arr / 255.0
        x_predict = img_arr[tf.newaxis, ...]
        print("x_predict:", x_predict.shape)
        result = model.predict(x_predict)
        result = np.squeeze(result)
        pred = np.argmax(result)
        print("结果：{}, 预测：{}, 置信值:{}".format(result, pred, result[pred]))
        print('\n')
        tf.print(pred)

if __name__ == '__main__':
    main()
