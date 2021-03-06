# Copyright 2020 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from unittest import TestCase

import pytest


@pytest.mark.gpu_test
class TestMiniCuda(TestCase):
    def test_check_cuda_error(self):
        from tests.model_compiler import mini_cuda  # pylint: disable=import-outside-toplevel

        mini_cuda.init()

        with contextlib.closing(mini_cuda.get_device(0).create_context(0)):
            memory = mini_cuda.allocate_memory(16)

        with self.assertRaises(RuntimeError) as context_manager:
            memory.close()

        self.assertEqual(context_manager.exception.args, ('CUDA error: invalid argument (1).',))

    @staticmethod
    def test_close_twice():
        from tests.model_compiler import mini_cuda  # pylint: disable=import-outside-toplevel

        mini_cuda.init()

        with contextlib.closing(mini_cuda.get_device(0).create_context(0)):
            memory = mini_cuda.allocate_memory(16)

            memory.close()
            memory.close()
