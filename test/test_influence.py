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


def nn_graph():
    # create graph
    x = tf.placeholder(tf.float32, name='x_placeholder')
    y = tf.placeholder(tf.int32, name='y_placeholder')

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([_dim_features, _classes], 'weight')
        b_fc1 = bias_variable([_classes], 'bias')
        op_fc1 = tf.add(tf.matmul(x, W_fc1), b_fc1)

    # set loss function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=op_fc1)
    cross_entropy = tf.reduce_mean(cross_entropy)
    return x, y, cross_entropy


class TestInfluence(unittest.TestCase):
    def setUp(self):
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

        x, y, cross_entropy = nn_graph()

        # open session
        self.sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint('test/data'))

        self.graph_origin = tf.get_default_graph().as_graph_def()
        # initialize influence function
        self.insp = darkon.Influence(workspace='./tmp',
                                     feeder=MyFeeder(),
                                     loss_op_train=cross_entropy,
                                     loss_op_test=cross_entropy,
                                     x_placeholder=x,
                                     y_placeholder=y)

    def tearDown(self):
        self.sess.close()

    # def test_freeze_graph(self):
    #     saver = tf.train.Saver()
    #     with tf.Session() as sess:
    #         # sess.run(tf.global_variables_initializer())
    #         saver.restore(sess, tf.train.latest_checkpoint('test/data-origin'))
    #         saver.save(sess, 'test/data/model', global_step=0)

    def test_influence(self):
        test_indices = [0]
        approx_params = {'scale': 10,
                         'num_repeats': 3,
                         'recursion_depth': 2,
                         'recursion_batch_size': _batch_size}

        # get influence scores for all trainset
        result = self.insp.upweighting_influence_batch(self.sess,
                                                       test_indices=test_indices,
                                                       test_batch_size=_batch_size,
                                                       approx_params=approx_params,
                                                       train_batch_size=_batch_size,
                                                       train_iterations=_num_iterations,
                                                       force_refresh=True)

        # get influence scores for all trainset
        result2 = self.insp.upweighting_influence_batch(self.sess,
                                                        test_indices=test_indices,
                                                        test_batch_size=_batch_size,
                                                        approx_params=approx_params,
                                                        train_batch_size=_batch_size,
                                                        train_iterations=_num_iterations,
                                                        force_refresh=False)

        self.assertEqual(_batch_size * _num_iterations, len(result2))
        self.assertTrue(np.all(result == result2))

        selected_trainset = [2, 3, 0, 9, 14, 19, 8]
        result_partial = self.insp.upweighting_influence(self.sess,
                                                         test_indices=test_indices,
                                                         test_batch_size=_batch_size,
                                                         approx_params=approx_params,
                                                         train_indices=selected_trainset,
                                                         num_total_train_example=_num_train_data,
                                                         force_refresh=False)
        self.assertEqual(7, len(result_partial))

    def test_influence_sampling(self):
        test_indices = [0]
        approx_batch_size = _batch_size
        approx_params = {'scale': 10,
                         'num_repeats': 3,
                         'recursion_depth': 2,
                         'recursion_batch_size': approx_batch_size}

        result = self.insp.upweighting_influence_batch(self.sess,
                                                       test_indices=test_indices,
                                                       test_batch_size=_batch_size,
                                                       approx_params=approx_params,
                                                       train_batch_size=_batch_size,
                                                       train_iterations=_num_iterations,
                                                       force_refresh=False)
        self.assertEqual(_batch_size * _num_iterations, len(result))

        num_batch_sampling = 2
        result2 = self.insp.upweighting_influence_batch(self.sess,
                                                        test_indices=test_indices,
                                                        test_batch_size=_batch_size,
                                                        approx_params=approx_params,
                                                        train_batch_size=_batch_size,
                                                        train_iterations=_num_iterations,
                                                        subsamples=num_batch_sampling,
                                                        force_refresh=False)
        self.assertEqual(num_batch_sampling * _num_iterations, len(result2))

        result = result.reshape(_num_iterations, _batch_size)
        result2 = result2.reshape(_num_iterations, num_batch_sampling)
        result = result[:, :num_batch_sampling]
        self.assertTrue(np.all(result == result2))

    def test_unknown_approx_key(self):
        test_indices = [0]
        approx_params = {'unknown_param': 1}
        self.assertRaises(RuntimeError,
                          self.insp.upweighting_influence_batch,
                          self.sess,
                          test_indices=test_indices,
                          test_batch_size=_batch_size,
                          approx_params=approx_params,
                          train_batch_size=_batch_size,
                          train_iterations=_num_iterations)

    def test_default_approx_params(self):
        test_indices = [0]
        r = self.insp.upweighting_influence_batch(self.sess,
                                                  test_indices=test_indices,
                                                  test_batch_size=_batch_size,
                                                  approx_params=None,
                                                  train_batch_size=_batch_size,
                                                  train_iterations=_num_iterations)

        r2 = self.insp.upweighting_influence_batch(self.sess,
                                                   test_indices,
                                                   _batch_size,
                                                   None,
                                                   _batch_size,
                                                   _num_iterations)
        self.assertTrue(np.all(r == r2))

    def test_approx_filename(self):
        test_indices = [0]
        approx_params = {'scale': 10,
                         'num_repeats': 3,
                         'recursion_depth': 2,
                         'recursion_batch_size': _batch_size}

        inv_hvp_filename = 'ihvp.c089c98599898bfb0e7f920c9dfe533af38b5481.npz'
        self.insp.ihvp_config.update(approx_params)
        self.assertEqual(inv_hvp_filename, self.insp._approx_filename(self.sess, test_indices))

        test_indices = [1]
        self.assertNotEqual(inv_hvp_filename, self.insp._approx_filename(self.sess, test_indices))

        test_indices = [0]
        self.insp.ihvp_config.update(scale=1)
        self.assertNotEqual(inv_hvp_filename, self.insp._approx_filename(self.sess, test_indices))

    def test_approx_filename_for_weight(self):
        test_indices = [0]

        filename_1 = self.insp._approx_filename(self.sess, test_indices)
        filename_2 = self.insp._approx_filename(self.sess, test_indices)
        self.assertEqual(filename_1, filename_2)

        self.sess.run(tf.global_variables_initializer())
        filename_3 = self.insp._approx_filename(self.sess, test_indices)
        self.assertNotEqual(filename_1, filename_3)

    def test_graph_dangling(self):
        test_indices = [0]
        approx_params = {'scale': 10,
                         'num_repeats': 3,
                         'recursion_depth': 2,
                         'recursion_batch_size': _batch_size}

        graph_influence_init = tf.get_default_graph().as_graph_def()
        self.assertNotEqual(self.graph_origin, graph_influence_init)

        self.insp.upweighting_influence(self.sess,
                                        test_indices=test_indices,
                                        test_batch_size=_batch_size,
                                        approx_params=approx_params,
                                        train_indices=[0],
                                        num_total_train_example=_num_train_data,
                                        force_refresh=True)

        graph_first_executed = tf.get_default_graph().as_graph_def()
        self.assertEqual(graph_influence_init, graph_first_executed)

        self.insp.upweighting_influence(self.sess,
                                        test_indices=test_indices,
                                        test_batch_size=_batch_size,
                                        approx_params=approx_params,
                                        train_indices=[0],
                                        num_total_train_example=_num_train_data,
                                        force_refresh=True)

        graph_second_executed = tf.get_default_graph().as_graph_def()
        self.assertEqual(graph_first_executed, graph_second_executed)
