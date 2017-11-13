"""Copyright 2017 Neosapience, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import unittest

import darkon
import tensorflow as tf
import numpy as np


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


_num_train_data = 20
_dim_features = 5
_num_test_data = 3
_classes = 2
_batch_size = 4
_trained_steps = 5


class ModuleCheck(unittest.TestCase):
    def setUp(self):
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

        # create graph
        x = tf.placeholder(tf.float32, name='x_placeholder')
        y = tf.placeholder(tf.int32, name='y_placeholder')

        with tf.name_scope('fc1'):
            W_fc1 = weight_variable([_dim_features, _classes], 'weight')
            b_fc1 = bias_variable([_classes], 'bias')
            op_fc1 = tf.add(tf.matmul(x, W_fc1), b_fc1)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=op_fc1)
        cross_entropy = tf.reduce_mean(cross_entropy)

        self.feeder_cls = MyFeeder
        self.loss_op = cross_entropy
        self.x = x
        self.y = y

    # def test_freeze_graph(self):
    #     saver = tf.train.Saver()
    #     with tf.Session() as sess:
    #         # sess.run(tf.global_variables_initializer())
    #         saver.restore(sess, tf.train.latest_checkpoint('test/data-origin'))
    #         saver.save(sess, 'test/data/model', global_step=0)

    def test_influence(self):
        # initialize influence function
        inspector = darkon.Influence(workspace='./tmp',
                                     feeder=self.feeder_cls(),
                                     loss_op_train=self.loss_op,
                                     loss_op_test=self.loss_op,
                                     x_placeholder=self.x,
                                     y_placeholder=self.y)

        self.assertNotEqual(None, inspector)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint('test/data'))

            # make inverse hessian vector
            test_indices = [0]
            approx_batch_size = _batch_size
            inspector.prepare(sess,
                              test_indices,
                              approx_params={'scale': 10,
                                             'num_repeats': 3,
                                             'recursion_depth': 2,
                                             'recursion_batch_size': approx_batch_size},
                              force_refresh=True,
                              test_batch_size=_batch_size)

            # get influence scores for all trainset
            result = inspector.upweighting_influence_batch(sess, _batch_size, _trained_steps)
            self.assertEqual(_batch_size * _trained_steps, len(result))

            # get influence scores for all trainset
            result2 = inspector.upweighting_influence_batch(sess, _batch_size, _trained_steps)
            self.assertEqual(_batch_size * _trained_steps, len(result2))
            self.assertTrue(np.all(result == result2))

            # get influence scores for selected trainset
            selected = [2, 3, 0, 9, 14, 19, 8]
            result_partial = inspector.upweighting_influence(sess, selected, _num_train_data)
            self.assertEqual(7, len(result_partial))

    def test_influence_sampling(self):
        # initialize influence function
        inspector = darkon.Influence(workspace='./tmp',
                                     feeder=self.feeder_cls(),
                                     loss_op_train=self.loss_op,
                                     loss_op_test=self.loss_op,
                                     x_placeholder=self.x,
                                     y_placeholder=self.y)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint('test/data'))

            # make inverse hessian vector
            test_indices = [0]
            approx_batch_size = _batch_size
            inspector.prepare(sess,
                              test_indices,
                              approx_params={'scale': 10,
                                             'num_repeats': 3,
                                             'recursion_depth': 2,
                                             'recursion_batch_size': approx_batch_size},
                              force_refresh=False,
                              test_batch_size=_batch_size)

            result = inspector.upweighting_influence_batch(sess, _batch_size, _trained_steps)
            self.assertEqual(_batch_size * _trained_steps, len(result))

            num_batch_sampling = 2
            result2 = inspector.upweighting_influence_batch(sess, _batch_size, _trained_steps, num_batch_sampling)
            self.assertEqual(num_batch_sampling * _trained_steps, len(result2))

            result = result.reshape(_trained_steps, _batch_size)
            result2 = result2.reshape(_trained_steps, num_batch_sampling)
            result = result[:, :num_batch_sampling]
            self.assertTrue(np.all(result == result2))

    def test_approx_filename(self):
        inspector = darkon.Influence(workspace='./tmp',
                                     feeder=self.feeder_cls(),
                                     loss_op_train=self.loss_op,
                                     loss_op_test=self.loss_op,
                                     x_placeholder=self.x,
                                     y_placeholder=self.y)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint('test/data'))

            test_indices = [0]
            approx_params = {'scale': 10,
                             'num_repeats': 3,
                             'recursion_depth': 2,
                             'recursion_batch_size': _batch_size}

            inv_hvp_filename = 'ihvp.c089c98599898bfb0e7f920c9dfe533af38b5481.npz'
            inspector.ihvp_config.update(approx_params)
            self.assertEqual(inv_hvp_filename, inspector._approx_filename(sess, test_indices))

            test_indices = [1]
            self.assertNotEqual(inv_hvp_filename, inspector._approx_filename(sess, test_indices))

            test_indices = [0]
            inspector.ihvp_config.update(scale=1)
            self.assertNotEqual(inv_hvp_filename, inspector._approx_filename(sess, test_indices))

    def test_unknown_approx_key(self):
        inspector = darkon.Influence(workspace='./tmp',
                                     feeder=self.feeder_cls(),
                                     loss_op_train=self.loss_op,
                                     loss_op_test=self.loss_op,
                                     x_placeholder=self.x,
                                     y_placeholder=self.y)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint('test/data'))

            test_indices = [0]
            self.assertRaises(RuntimeError,
                              inspector.prepare,
                              sess,
                              test_indices,
                              approx_params={'unknown_param': 1},
                              test_batch_size=_batch_size)

    def test_invalid_call(self):
        inspector = darkon.Influence(workspace='./tmp',
                                     feeder=self.feeder_cls(),
                                     loss_op_train=self.loss_op,
                                     loss_op_test=self.loss_op,
                                     x_placeholder=self.x,
                                     y_placeholder=self.y)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint('test/data'))

            self.assertRaises(RuntimeError,
                              inspector.upweighting_influence_batch,
                              sess,
                              _batch_size,
                              _trained_steps
                              )
