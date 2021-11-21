import netron
# modelPath = "./frozen_model/model.pb"
# modelPath = "./model_save/saved_model.pb"
# modelPath = "./h5_model/model.h5"
modelPath = "./tflite_unquant/converted_from_unquant_save_model.tflite"
# modelPath = "./tflite_quant/converted_from_quant_weight.tflite"

netron.start(modelPath)