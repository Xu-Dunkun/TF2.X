import numpy as np
import tensorflow as tf
import os
from PIL import Image
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    global tflite_model_file
    pre_img_path = '/modelmain/img_data/img_pre'
    height = 228
    width = 228
    pre_tflite_flag = 0

    if pre_tflite_flag == 0:
        tflite_model_file = './tflite_unquant/unquant_h5.tflite'
    elif pre_tflite_flag == 1:
        tflite_model_file = './tflite_quant/quant_weight.tflite'
    elif pre_tflite_flag == 2:
        tflite_model_file = './tflite_quant/quant_full_integer.tflite'
    elif pre_tflite_flag == 3:
        tflite_model_file = './tflite_quant/quant_only_integer.tflite'

    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
    interpreter.allocate_tensors()

    prediction(interpreter, pre_img_path, height, width)


def prediction(interpreter, pre_img_path, height, width):
    for img_name in os.listdir(pre_img_path):
        pre_img_path = pre_img_path + '/' + img_name
        img = Image.open(pre_img_path)
        img = img.resize((height, width), Image.ANTIALIAS)
        img_arr = np.array(img.convert('L'))
        img_arr = img_arr / 255.0
        pre_image = np.expand_dims(img_arr, axis=0).astype(np.float32)
        print("x_predict:", pre_image.shape)
        # pre_image = img_arr[tf.newaxis, ...]

        start = time.clock()
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        interpreter.set_tensor(input_index, pre_image)
        interpreter.invoke()
        output_result = interpreter.get_tensor(output_index)

        pre_digit = np.argmax(output_result()[0])
        finish = time.clock()
        pre_time = (finish - start) * 1000
        print('-' * 25)
        print("结果：{}".format(output_result()[0]))
        print("预测：{}, 置信值:{}".format(pre_digit, output_result()[0][pre_digit]))
        print("耗时：%4.f " % pre_time + ' ms')
        print('-' * 25)
        print('\n')
