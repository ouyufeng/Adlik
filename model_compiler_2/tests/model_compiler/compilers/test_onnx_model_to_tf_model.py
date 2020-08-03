# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from tempfile import NamedTemporaryFile
from unittest import TestCase

import model_compiler.compilers.tf_model_to_tf_frozen_graph_model as tf_model_compiler
import model_compiler.compilers.tf_frozen_graph_model_to_onnx_model as frozen_graph_compiler
import model_compiler.compilers.onnx_model_to_tf_model as onnx_to_tf_compiler
import model_compiler.compilers.onnx_model_file_to_onnx_model as onnx_compiler

from model_compiler.compilers.onnx_model_file_to_onnx_model import Config
from model_compiler.models.irs.tf_model import Input, TensorFlowModel
from model_compiler.models.sources.onnx_model_file import ONNXModelFile

import tensorflow as tf
import onnx


def _make_onnx_model():
    with tf.Graph().as_default(), tf.compat.v1.Session().as_default() as session:
        input_x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[3, 4], name='x')
        input_y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[3, 4], name='y')
        weight = tf.Variable(initial_value=4.2, dtype=tf.float32)
        output_z = tf.multiply(input_x + input_y, weight, name='z')

        session.run(weight.initializer)

    frozen_graph_model = tf_model_compiler.compile_source(
        source=TensorFlowModel(inputs=[Input(tensor=input_x), Input(tensor=input_y)],
                               outputs=[output_z],
                               session=session)
    )

    return frozen_graph_compiler.compile_source(frozen_graph_model)


class CompileSourceTestCase(TestCase):
    def test_compile_with_variables(self):
        onnx_model = _make_onnx_model()
        compiled = onnx_to_tf_compiler.compile_source(source=onnx_model)

        self.assertIsInstance(compiled.session, tf.compat.v1.Session)
        self.assertEqual([model_input.tensor.name[0] for model_input in compiled.inputs], ['x', 'y'])
        self.assertEqual([model_input.data_format for model_input in compiled.inputs], [None, None])
        self.assertEqual([model_output.name[0] for model_output in compiled.outputs], ['z'])

    def test_with_no_data_formats(self):
        with NamedTemporaryFile(suffix='.onnx') as model_file:
            onnx.save_model(_make_onnx_model().model_proto, model_file.name)

            config = Config.from_json({})
            onnx_model = onnx_compiler.compile_source(source=ONNXModelFile(model_file.name), config=config)

        compiled = onnx_to_tf_compiler.compile_source(source=onnx_model)

        self.assertIsInstance(compiled.session, tf.compat.v1.Session)
        self.assertEqual([model_input.data_format for model_input in compiled.inputs], [None, None])
