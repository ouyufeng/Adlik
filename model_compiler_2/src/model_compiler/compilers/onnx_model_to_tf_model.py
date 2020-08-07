# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
from onnx_tf.backend import prepare

from . import repository
from ..models.irs.onnx_model import OnnxModel
from ..models.irs.tf_model import Input, TensorFlowModel


@repository.REPOSITORY.register(source_type=OnnxModel, target_type=TensorFlowModel)
def compile_source(source: OnnxModel) -> TensorFlowModel:
    tf_representation = prepare(source.model_proto)
    input_tensors = [tf_representation.tensor_dict[input_name] for input_name in tf_representation.inputs]

    return TensorFlowModel(inputs=[Input(input_tensor, data_format)
                                   for input_tensor, data_format in
                                   zip(input_tensors, list(source.input_data_formats))],
                           outputs=[tf_representation.tensor_dict[output_name]
                                    for output_name in tf_representation.outputs],
                           session=tf.compat.v1.Session(graph=tf_representation.graph))
