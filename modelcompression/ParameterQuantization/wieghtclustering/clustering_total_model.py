import tensorflow_model_optimization as tfmot
import tensorflow as tf
import pathlib

def main():

    dirname = './tflite_weight_clustering'
    filename = 'clustering_total_model.tflite'
    train_images = '/modelmain/data/img_data'
    train_labels = '/modelmain/data/img_label'

    model = tf.keras.models.load_model('trainandsave/model_h5/model.h5')

    cluster_weights = tfmot.clustering.keras.cluster_weights
    CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

    clustering_params = {
        'number_of_clusters': 16,
        'cluster_centroids_init': CentroidInitialization.LINEAR
    }

    clustered_model = cluster_weights(model, **clustering_params)

    # Use smaller learning rate for fine-tuning clustered model
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
    clustered_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            optimizer=opt,
                            metrics=['accuracy'])

    clustered_model.summary()

    # Fine-tune model
    clustered_model.fit(train_images, train_labels, batch_size=500, epochs=1, validation_split=0.1)

    converter = tf.lite.TFLiteConverter.from_keras_model(clustered_model)
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
