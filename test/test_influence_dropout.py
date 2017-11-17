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
from .tf_util import weight_variable, bias_variable


_num_train_data = 20
_dim_features = 5
_num_test_data = 3
_classes = 2
_batch_size = 4
_num_iterations = 5


def nn_graph_dropout():
    # create graph
    x = tf.placeholder(tf.float32, name='x_placeholder')
    y = tf.placeholder(tf.int32, name='y_placeholder')

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([_dim_features, _classes], 'weight')
        b_fc1 = bias_variable([_classes], 'bias')
        op_fc1 = tf.add(tf.matmul(x, W_fc1), b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        drop_fc1 = tf.nn.dropout(op_fc1, keep_prob)

    # set loss function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=drop_fc1)
    cross_entropy = tf.reduce_mean(cross_entropy)
    return x, y, cross_entropy, keep_prob


class TestInfluenceWithDropout(unittest.TestCase):
    def tearDown(self):
        # init tf default graph
        tf.reset_default_graph()

        # dataset feeder
        class MyFeeder(darkon.InfluenceFeeder):
            def __init__(self):
                self.train_x = np.random.uniform(size=_num_train_data * _dim_features).reshape([_num_train_data, -1])
                self.train_y = np.random.randint(_classes, size=_num_train_data).reshape([-1])
                self.test_x = np.random.uniform(size=_num_test_data * _dim_features).reshape([_num_test_data, -1])
                self.test_y = np.random.randint(_classes, size=_num_test_data).reshape([-1])

                self.train_y = np.eye(_classes)[self.train_y]
                self.test_y = np.eye(_classes)[self.test_y]

            def reset(self):
                np.random.seed(97)

            def train_batch(self, batch_size):
                idx = np.random.choice(_num_train_data - batch_size + 1, 1)[0]
                return self.train_x[idx:idx+batch_size], self.train_y[idx:idx+batch_size]

            def train_one(self, index):
                return self.train_x[index], self.train_y[index]

            def test_indices(self, indices):
                return self.test_x[indices], self.test_y[indices]

        x, y, cross_entropy, keep_prob = nn_graph_dropout()

        self.insp = darkon.Influence(workspace='./tmp',
                                     feeder=MyFeeder(),
                                     loss_op_train=cross_entropy,
                                     loss_op_test=cross_entropy,
                                     x_placeholder=x,
                                     y_placeholder=y,
                                     test_feed_options={keep_prob: 1.0},
                                     train_feed_options={keep_prob: self.train_keep_prob})
        # open session
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint('test/data'))

            test_indices = [0]
            approx_params = {'scale': 10,
                             'num_repeats': 3,
                             'recursion_depth': 2,
                             'recursion_batch_size': _batch_size}

            # get influence scores for all trainset
            result = self.insp.upweighting_influence_batch(sess,
                                                           test_indices=test_indices,
                                                           test_batch_size=_batch_size,
                                                           approx_params=approx_params,
                                                           train_batch_size=_batch_size,
                                                           train_iterations=_num_iterations,
                                                           force_refresh=True)

            result2 = self.insp.upweighting_influence_batch(sess,
                                                            test_indices=test_indices,
                                                            test_batch_size=_batch_size,
                                                            approx_params=approx_params,
                                                            train_batch_size=_batch_size,
                                                            train_iterations=_num_iterations,
                                                            force_refresh=False)

            self.assertEqual(_batch_size * _num_iterations, len(result2))

            # use dropout or not
            if 1.0 > self.train_keep_prob:
                self.assertFalse(np.all(result == result2))
            else:
                self.assertTrue(np.all(result == result2))

    def test_with_dropout(self):
        self.train_keep_prob = 0.5

    def test_without_dropout(self):
        self.train_keep_prob = 1.0
