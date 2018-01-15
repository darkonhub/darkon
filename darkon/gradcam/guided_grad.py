# Copyright 2017 Neosapience, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_grad
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.util import compat
from tensorflow.contrib.graph_editor import subgraph


_grad_override_map = {
    'Tanh': 'GuidedTanh',
    'Sigmoid': 'GuidedSigmoid',
    'Relu': 'GuidedRelu',
    'Relu6': 'GuidedRelu6',
    'Elu': 'GuidedElu',
    'Selu': 'GuidedSelu',
    'Softplus': 'GuidedSoftplus',
    'Softsign': 'GuidedSoftsign',
}


def replace_grad_to_guided_grad(g):
    sgv = subgraph.make_view(g)
    with g.gradient_override_map(_grad_override_map):
        for op in sgv.ops:
            _replace_grad(g, op)


def _replace_grad(g, op):
    # ref: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/ops.py
    # tf.Graph._gradient_override_map
    try:
        op_def = op._op_def
        node_def = op._node_def

        if op_def is not None:
            mapped_op_type = g._gradient_override_map[op_def.name]
            node_def.attr["_gradient_op_type"].CopyFrom(
                attr_value_pb2.AttrValue(s=compat.as_bytes(mapped_op_type)))
    except KeyError:
        pass


def guided_grad(grad):
    return tf.where(0. < grad, grad, tf.zeros_like(grad))


@ops.RegisterGradient("GuidedTanh")
def _guided_grad_tanh(op, grad):
    return guided_grad(math_grad._TanhGrad(op, grad))


@ops.RegisterGradient("GuidedSigmoid")
def _guided_grad_sigmoid(op, grad):
    return guided_grad(math_grad._SigmoidGrad(op, grad))


@ops.RegisterGradient("GuidedRelu")
def _guided_grad_relu(op, grad):
    return guided_grad(gen_nn_ops._relu_grad(grad, op.outputs[0]))


@ops.RegisterGradient("GuidedRelu6")
def _guided_grad_relu6(op, grad):
    return guided_grad(gen_nn_ops._relu6_grad(grad, op.outputs[0]))


@ops.RegisterGradient("GuidedElu")
def _guided_grad_elu(op, grad):
    return guided_grad(gen_nn_ops._elu_grad(grad, op.outputs[0]))


@ops.RegisterGradient("GuidedSelu")
def _guided_grad_selu(op, grad):
    return guided_grad(gen_nn_ops._selu_grad(grad, op.outputs[0]))


@ops.RegisterGradient("GuidedSoftplus")
def _guided_grad_softplus(op, grad):
    return guided_grad(gen_nn_ops._softplus_grad(grad, op.outputs[0]))


@ops.RegisterGradient("GuidedSoftsign")
def _guided_grad_softsign(op, grad):
    return guided_grad(gen_nn_ops._softsign_grad(grad, op.outputs[0]))
