import tensorflow_model_optimization as tfmot
import tensorflow as tf
import pathlib

def main():

    dirname = './tflite_quant_aware'
    filename = 'quant_total_model.tflite'
    train_images = '/modelmain/data/img_data'
    train_labels = '/modelmain/data/img_label'

    quantize_model = tfmot.quantization.keras.quantize_model

    model = tf.keras.models.load_model('trainandsave/model_h5/model.h5')

    # q_aware stands for for quantization aware.
    q_aware_model = quantize_model(model)

    # `quantize_model` requires a recompile.
    q_aware_model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])

    q_aware_model.summary()

    train_images_subset = train_images[0:1000]
    train_labels_subset = train_labels[0:1000]

    q_aware_model.fit(train_images_subset, train_labels_subset,
                      batch_size=500, epochs=1, validation_split=0.1)

    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    tflite_model = converter.convert()
    save_tflite_model(tflite_model, dirname, filename)

def save_tflite_model(tflite_model, dirname, filename):
    tflite_models_dir = pathlib.Path(dirname)
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir + '/' + filename
    tflite_model_file.write_bytes(tflite_model)
    pass

if __name__ == "__main__":
    main()