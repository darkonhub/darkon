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

    with tf.name_scope('logits'):
        logits = tf.nn.softmax(top)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        cross_entropy = tf.reduce_mean(cross_entropy)

    return x, y, cross_entropy


class GradcamUtil(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def tearDown(self):
        pass

    def test_relu(self):
        x, y, cross_entropy = nn_graph(activation=tf.nn.relu)
        image = np.random.uniform(size=(2, 2, 3))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            gradcam_ops = darkon.Gradcam.candidate_featuremap_op_names(sess)
            prob_ops = darkon.Gradcam.candidate_predict_op_names(sess, _classes)

            insp = darkon.Gradcam(x, _classes, gradcam_ops[-1], prob_ops[-1])
            ret = insp.gradcam(sess, image)
            self.assertTrue(ret)
