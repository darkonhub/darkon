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
import tensorflow as tf
import numpy as np

_unusable_op_names = ('Shape', 'Reshape', 'Slice', 'Pack', 'Cast', 'ConcatV2')


def _unusable_ops(op):
    if len(op.outputs) == 0 \
            or 'save' in op.name \
            or op.op_def is None \
            or op.op_def.name in _unusable_op_names:
        return True
    else:
        return False


def candidate_featuremap_op_names(sess, graph):
    operations = []
    out_ranks = []
    out_shapes = []

    for op in graph.get_operations():
        if _unusable_ops(op):
            continue

        out_ranks.append(tf.rank(op.outputs[0]))
        out_shapes.append(tf.shape(op.outputs[0]))
        operations.append(op)

    out_ranks_val, out_shapes_val = sess.run([out_ranks, out_shapes])

    ret = []
    for out_rank, out_shape, op in zip(out_ranks_val, out_shapes_val, operations):
        if out_rank != 4 or out_shape[1] == 1 or out_shape[2] == 1 or out_shape[0] != 1:
            continue

        ret.append(op.name)
    return ret


def candidate_predict_op_names(sess, num_classes, graph):
    operations = []
    out_shapes = []

    for op in graph.get_operations():
        if _unusable_ops(op):
            continue

        out_shapes.append(tf.shape(op.outputs[0]))
        operations.append(op)

    out_shapes_val = sess.run(out_shapes)

    ret = []
    for out_shape, op in zip(out_shapes_val, operations):
        if np.prod(out_shape) != num_classes:
            continue

        ret.append(op.name)
    return ret
