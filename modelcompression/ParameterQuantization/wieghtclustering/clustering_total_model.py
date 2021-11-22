import tensorflow_model_optimization as tfmot
import tensorflow as tf

train_images = 1
train_labels = 2

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
