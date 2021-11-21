import onnx
from onnx_tf.backend import prepare

onnx_input_path = 'shuffle2_sm_sim.onnx'
pb_output_path = 'shuffle2_sm_sim'
onnx_model = onnx.load(onnx_input_path)  # load onnx model
tf_exp = prepare(onnx_model)  # prepare tf representation
tf_exp.export_graph(pb_output_path)  # export the mo


