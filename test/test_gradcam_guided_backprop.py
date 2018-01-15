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
import unittest

import darkon
import tensorflow as tf
import numpy as np

_classes = 2


def nn_graph(activation):
    # create graph
    x = tf.placeholder(tf.float32, (1, 2, 2, 3), 'x_placeholder')
    y = tf.placeholder(tf.int32, name='y_placeholder', shape=[1, 2])

    with tf.name_scope('conv1'):
        conv_1 = tf.layers.conv2d(
            inputs=x,
            filters=10,
            kernel_size=[2, 2],
            padding="same",
            activation=activation)

    with tf.name_scope('fc2'):
        flatten = tf.layers.flatten(conv_1)
        top = tf.layers.dense(flatten, _classes)

    logits = tf.nn.softmax(top)
    return x


class GradcamGuidedBackprop(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def tearDown(self):
        x = nn_graph(activation=self.activation_fn)
        image = np.random.uniform(size=(2, 2, 3))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            gradcam_ops = darkon.Gradcam.candidate_featuremap_op_names(sess)

            if self.enable_guided_backprop:
                _ = darkon.Gradcam(x, _classes, gradcam_ops[-1])

            g = tf.get_default_graph()
            from_ts = g.get_operation_by_name(gradcam_ops[-1]).outputs
            to_ts = g.get_operation_by_name(gradcam_ops[-2]).outputs

            max_output = tf.reduce_max(from_ts, axis=3)
            y = tf.reduce_sum(-max_output * 1e2)

            grad = tf.gradients(y, to_ts)[0]
            grad_val = sess.run(grad, feed_dict={x: np.expand_dims(image, 0)})

            if self.enable_guided_backprop:
                self.assertTrue(not np.any(grad_val))
            else:
                self.assertTrue(np.any(grad_val))

    def test_relu(self):
        self.activation_fn = tf.nn.relu
        self.enable_guided_backprop = False

    def test_relu_guided(self):
        self.activation_fn = tf.nn.relu
        self.enable_guided_backprop = True

    def test_tanh(self):
        self.activation_fn = tf.nn.tanh
        self.enable_guided_backprop = False

    def test_tanh_guided(self):
        self.activation_fn = tf.nn.tanh
        self.enable_guided_backprop = True

    def test_sigmoid(self):
        self.activation_fn = tf.nn.sigmoid
        self.enable_guided_backprop = False

    def test_sigmoid_guided(self):
        self.activation_fn = tf.nn.sigmoid
        self.enable_guided_backprop = True

    def test_relu6(self):
        self.activation_fn = tf.nn.relu6
        self.enable_guided_backprop = False

    def test_relu6_guided(self):
        self.activation_fn = tf.nn.relu6
        self.enable_guided_backprop = True

    def test_elu(self):
        self.activation_fn = tf.nn.elu
        self.enable_guided_backprop = False

    def test_elu_guided(self):
        self.activation_fn = tf.nn.elu
        self.enable_guided_backprop = True

    def test_selu(self):
        self.activation_fn = tf.nn.selu
        self.enable_guided_backprop = False

    def test_selu_guided(self):
        self.activation_fn = tf.nn.selu
        self.enable_guided_backprop = True

    def test_softplus(self):
        self.activation_fn = tf.nn.softplus
        self.enable_guided_backprop = False

    def test_test_softplus_guided(self):
        self.activation_fn = tf.nn.softplus
        self.enable_guided_backprop = True

    def test_softsign(self):
        self.activation_fn = tf.nn.softsign
        self.enable_guided_backprop = False

    def test_softsign_guided(self):
        self.activation_fn = tf.nn.softsign
        self.enable_guided_backprop = True
