import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import time

def preNum(interpreter, input_tensor_index, output, prediction_digits, test_labels):
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
        x_predict = tf.cast(x_predict, tf.float32)
        print("x_predict:", x_predict.shape)

        interpreter.set_tensor(input_tensor_index, x_predict)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)

        # Compare prediction results with ground truth labels to calculate accuracy.
        accurate_count = 0
        for index in range(len(prediction_digits)):
            if prediction_digits[index] == test_labels[index]:
                accurate_count += 1
        accuracy = accurate_count * 1.0 / len(prediction_digits)

        return accuracy

# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_tflite_model(tflite_model_path):
    # Initialize TFLite interpreter using the model.
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_tensor_index = interpreter.get_input_details()[0]["index"]
    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []

    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # preNum(interpreter, input_tensor_index, output, prediction_digits)

    for test_image in test_images:
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        test_image = tf.reshape(test_image, (1, 28, 28, 1))
        interpreter.set_tensor(input_tensor_index, test_image)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)

        # Compare prediction results with ground truth labels to calculate accuracy.
        accurate_count = 0
        for index in range(len(prediction_digits)):
            if prediction_digits[index] == test_labels[index]:
                accurate_count += 1
        accuracy = accurate_count * 1.0 / len(prediction_digits)

        return accuracy


# Evaluate the TF Lite float model. You'll find that its accurary is identical
# to the original TF (Keras) model because they are essentially the same model
# stored in different format.
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

time1 = time.time()

tflite_quantized_model_path = './tflite_quant/quant_weight_variables_activation.tflite'

quantized_accuracy = evaluate_tflite_model(tflite_quantized_model_path)
print('Quantized model accuracy = %.4f' % quantized_accuracy)
time2 = time.time()
t = (time2-time1)*1000./int(test_images.shape[0])
print('Quantized model interpreter time = %.4f' % t + 'ms')

# Evalualte the TF Lite quantized model.
# Don't be surprised if you see quantized model accuracy is higher than
# the original float model. It happens sometimes :)

time3 = time.time()
tflite_float_model_path = './tflite_unquant/converted_from_unquant_save_model.tflite'
float_accuracy = evaluate_tflite_model(tflite_float_model_path)
print('Float model accuracy = %.4f' % quantized_accuracy)
time4 = time.time()
t = (time4-time3)*1000./int(test_images.shape[0])
print('Float model interpreter time = %.4f' % t + 'ms')


print('-'*25)
print('Accuracy drop = %.4f' % (float_accuracy - quantized_accuracy))


