import netron
# modelPath = "./frozen_model/mnist_model.pb"
# modelPath = "./mnist_model_save/saved_model.pb"
modelPath = "./mnist_h5_model/mnist_model.h5"
netron.start(modelPath)