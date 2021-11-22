import time
import numpy as np
import tensorflow as tf

def main():
    global tflite_model_file
    pre_img_path = '/modelmain/img_data/img_pre'
    height = 228
    width = 228
    pre_tflite_flag = 0

    if pre_tflite_flag == 0:
        tflite_model_file = './tflite_unquant/unquant_h5.tflite'
        evaluate_model(interpreter, test_images, test_labels, height, width)
    elif pre_tflite_flag == 1:
        tflite_model_file = './tflite_quant/quant_weight.tflite'
    elif pre_tflite_flag == 2:
        tflite_model_file = './tflite_quant/quant_full_integer.tflite'
    elif pre_tflite_flag == 3:
        tflite_model_file = './tflite_quant/quant_only_integer.tflite'

    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
    interpreter.allocate_tensors()


    test_images =
    test_labels =
    evaluate_model(interpreter, test_images, test_labels, height, width)

def evaluate_model(interpreter, test_images, test_labels, height, width):
    start = time.clock()
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

    print('-' * 25)
    print("模型准确率：%4.f " % accuracy)
    print("模型耗时：%4.f " % evaluate_time + ' ms')
    print('-' * 25)

if __name__ == '__main__':
    main()