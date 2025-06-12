from optimum.exporters.onnx import main_export

model_dir = "checkpoint-85000"
onnx_output_dir = "onnx_model_85000"

main_export(
    model_name_or_path=model_dir,
    output=onnx_output_dir,
    task="seq2seq-lm",
    inputs_as_int32=True,
)

import onnx
from onnx import TensorProto

encoder = onnx.load("onnx_model_85000/encoder_model.onnx")
decoder = onnx.load("onnx_model_85000/decoder_model.onnx")

for input_tensor in encoder.graph.input:
    if input_tensor.type.tensor_type.elem_type == TensorProto.INT64:
        input_tensor.type.tensor_type.elem_type = TensorProto.INT32
for input_tensor in decoder.graph.input:
    if input_tensor.type.tensor_type.elem_type == TensorProto.INT64:
        input_tensor.type.tensor_type.elem_type = TensorProto.INT32

onnx.save(encoder, "onnx_model_85000/encoder_model32.onnx")
onnx.save(decoder, "onnx_model_85000/decoder_model32.onnx")