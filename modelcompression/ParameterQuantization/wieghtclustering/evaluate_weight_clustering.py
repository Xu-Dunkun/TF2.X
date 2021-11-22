import time
import numpy as np
import tensorflow as tf
import os

def main():
    global tflite_model_file

    quant_aware_tflite_flag1 = 1
    quant_aware_tflite_flag2 = 2

    test_images = '/modelmain/data/img_data'
    test_labels = '/modelmain/data/img_label'
    height = 228
    width = 228
    pre_tflite_flag = 3

    if (pre_tflite_flag & quant_aware_tflite_flag1) == quant_aware_tflite_flag1:
        tflite_model_file = './tflite_weight_aware/quant_aware_model.tflite'
        evaluate_tflite_model(tflite_model_file, test_images, test_labels, height, width)
    if (pre_tflite_flag & quant_aware_tflite_flag2) == quant_aware_tflite_flag2:
        tflite_model_file = './tflite_quant_aware/quant_aware_layer.tflite'
        evaluate_tflite_model(tflite_model_file, test_images, test_labels, height, width)

def evaluate_tflite_model(tflite_model_file, test_images, test_labels, height, width):
    start = time.clock()
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    prediction_digits = []
    for test_image in test_images:
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        test_image = test_image / 255.0
        test_image = tf.reshape(test_image, (1, height, width, 3))
        interpreter.set_tensor(input_index, test_image)

        interpreter.invoke()

        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)

    finish = time.clock()
    evaluate_time = (finish - start) / len(test_images) * 1000

    accurate_count = 0
    for index in range(len(prediction_digits)):
        if prediction_digits[index] == test_labels[index]:
            accurate_count += 1
    accuracy = accurate_count * 1.0 / len(prediction_digits)

    model_size = os.path.getsize(tflite_model_file) / float(2 ** 20)

    print('-' * 25)
    print("model accuracy：%4.f " % accuracy)
    print("model inference time：%4.f " % evaluate_time + ' ms')
    print('model size：%.f ' % model_size + ' MB')
    print('-' * 25)

if __name__ == '__main__':
    main()