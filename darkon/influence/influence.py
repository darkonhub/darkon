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
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from tensorflow.python.ops.gradients_impl import _hessian_vector_product
from .feeder import InfluenceFeeder

import numpy as np
import tensorflow as tf

import os
import time
import hashlib
import json

_using_fully_tf = True


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('* %s function took [%.3fs]' % (f.__name__, time2-time1))
        return ret
    return wrap


# TODO: all of print function must be replaced to Logging function
class Influence:
    def __init__(self, workspace, feeder, loss_op_train, loss_op_test, x_placeholder, y_placeholder,
                 feed_options={}, trainable_variables=None):
        """ Create Influence instance

        Parameters
        ----------
        workspace: str
            path for workspace directory
        feeder : InfluenceFeeder
            dataset feeder
        loss_op_train : tf.Operation
            tensor for loss function
        loss_op_test : tf.Operation
            tensor for loss function
        x_placeholder : tf.Tensor
            tensor from tf.placeholder()
        y_placeholder : tf.Tensor
            tensor from tf.placeholder()
        feed_options : dict
            optional parameters to run loss operation
        trainable_variables : tuple, or list
            description
        """
        self.workspace = workspace
        self.feeder = feeder
        self.x_placeholder = x_placeholder
        self.y_placeholder = y_placeholder
        self.feed_options = feed_options

        if trainable_variables is None:
            trainable_variables = (
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) +
                tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))

        self.loss_op_train = loss_op_train
        self.grad_op_train = tf.gradients(loss_op_train, trainable_variables)
        self.grad_op_test = tf.gradients(loss_op_test, trainable_variables)

        self.v_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape()) for a in trainable_variables]
        self.hessian_vector_op = _hessian_vector_product(loss_op_train, trainable_variables, self.v_placeholder)
        self.inverse_hvp = None
        self.trainable_variables = trainable_variables

        self.ihvp_config = {
            'scale': 1e4,
            'damping': 0.01,
            'num_repeats': 1,
            'recursion_batch_size': 64,
            'recursion_depth': 10000
        }

        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)

    def _make_feed_dict(self, xs, ys):
        ret = {
            self.x_placeholder: xs,
            self.y_placeholder: ys,
        }
        ret.update(**self.feed_options)
        return ret

    def _path(self, *paths):
        return os.path.join(self.workspace, *paths)

    @timing
    def prepare(self, sess, test_indices, test_batch_size=64, approx_params=None, force_refresh=False):
        """ Calculate inverse hessian vector, and save it in workspace

        Parameters
        ----------
        sess: tf.Session
            desc
        test_indices: list
            desc
        approx_params: dict
            desc
        force_refresh: bool
            desc
        test_batch_size: int
            desc

        """
        self.feeder.reset()
        test_grad_loss = self._get_test_grad_loss(sess, test_indices, test_batch_size)
        print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate([a.reshape(-1) for a in test_grad_loss])))

        # update ihvp approx params
        for param_key in approx_params.keys():
            if param_key not in self.ihvp_config:
                raise RuntimeError('unknown ihvp config param is approx_params')
        self.ihvp_config.update(approx_params)
        inv_hvp_path = self._path(self._approx_filename(sess, test_indices))

        self._load_inverse_hvp(sess, force_refresh, test_grad_loss, inv_hvp_path)

    @timing
    def upweighting_influence(self, sess, train_indices, num_total_train_example):
        """ Measure loss different

        Parameters
        ----------
        sess: tf.Session
            desc
        train_indices: list
            desc
        num_total_train_example: int
            desc

        Returns
        -------
        array : np.ndarray

        """
        self.feeder.reset()
        predicted_loss_diffs = self._grad_diffs(sess, train_indices, num_total_train_example)
        print('Multiplying by %s train examples' % predicted_loss_diffs.size)
        return predicted_loss_diffs

    @timing
    def upweighting_influence_batch(self, sess, train_batch_size, num_iters, num_subsampling=-1):
        """ Measure loss different

        Parameters
        ----------
        sess: tf.Session
            desc
        train_batch_size: int
            desc
        num_iters: int
            desc
        num_subsampling: int
            desc

        Returns
        -------
        array : np.ndarray

        """
        if self.inverse_hvp == None:
            raise RuntimeError('You must call prepare() before')

        self.feeder.reset()
        predicted_loss_diffs = self._grad_diffs_all(sess, train_batch_size, num_iters, num_subsampling)
        print('Multiplying by %s train examples' % predicted_loss_diffs.size)
        return predicted_loss_diffs

    def _get_test_grad_loss(self, sess, test_indices, test_batch_size):
        if test_indices is not None:
            num_iter = int(np.ceil(len(test_indices) / test_batch_size))
            test_grad_loss = None
            for i in range(num_iter):
                start = i * test_batch_size
                end = int(min((i + 1) * test_batch_size, len(test_indices)))
                size = float(end - start)

                test_feed_dict = self._make_feed_dict(*self.feeder.test_indices(test_indices[start:end]))
                temp = sess.run(self.grad_op_test, feed_dict=test_feed_dict)
                temp = np.asarray(temp)

                temp *= size
                if test_grad_loss is None:
                    test_grad_loss = temp
                else:
                    test_grad_loss += temp

            test_grad_loss /= len(test_indices)
        else:
            raise RuntimeError('unsupported yet')
        return test_grad_loss

    def _load_inverse_hvp(self, sess, force_refresh, test_grad_loss, inv_hvp_path):
        if os.path.exists(inv_hvp_path) and not force_refresh:
            self.inverse_hvp = list(np.load(inv_hvp_path)['inverse_hvp'])
            print('Loaded inverse HVP from %s' % inv_hvp_path)
        else:
            self.inverse_hvp = self._get_inverse_hvp_lissa(sess, test_grad_loss)
            np.savez(inv_hvp_path, inverse_hvp=self.inverse_hvp)
            print('Saved inverse HVP to %s' % inv_hvp_path)

    def _approx_filename(self, sess, test_indices):
        sha = hashlib.sha1()

        # weights
        vs = sess.run(self.trainable_variables)
        for a in vs:
            sha.update(a.data)

        # test_indices
        np_test_indices = np.array(list(test_indices))
        sha.update(np_test_indices.data)

        # approx_params
        sha.update(json.dumps(self.ihvp_config, sort_keys=True).encode('utf-8'))
        return 'ihvp.' + sha.hexdigest() + '.npz'

    def _get_inverse_hvp_lissa(self, sess, v):
        ihvp_config = self.ihvp_config
        print_iter = ihvp_config['recursion_depth'] / 10

        if _using_fully_tf:
            estimation_op = self._estimation_op(v)

        inverse_hvp = None
        for sample_idx in range(ihvp_config['num_repeats']):
            cur_estimate = v
            # debug_diffs_estimation = []
            # prev_estimation_norm = np.linalg.norm(np.concatenate([a.reshape(-1) for a in cur_estimate]))

            for j in range(ihvp_config['recursion_depth']):
                train_batch_data, train_batch_label = self.feeder.train_batch(ihvp_config['recursion_batch_size'])
                feed_dict = self._make_feed_dict(train_batch_data, train_batch_label)
                feed_dict = self._update_feed_dict(feed_dict, cur_estimate)

                if _using_fully_tf:
                    cur_estimate = sess.run(estimation_op, feed_dict=feed_dict)
                else:
                    hessian_vector_val = sess.run(self.hessian_vector_op, feed_dict=feed_dict)
                    hessian_vector_val = np.array(hessian_vector_val)
                    cur_estimate = v + (1 - ihvp_config['damping']) * cur_estimate - hessian_vector_val / ihvp_config['scale']

                # curr_estimation_norm = np.linalg.norm(np.concatenate([a.reshape(-1) for a in cur_estimate]))
                # debug_diffs_estimation.append(curr_estimation_norm - prev_estimation_norm)
                # prev_estimation_norm = curr_estimation_norm

                if (j % print_iter == 0) or (j == ihvp_config['recursion_depth'] - 1):
                    print("Recursion at depth %s: norm is %.8lf" %
                          (j, np.linalg.norm(np.concatenate([a.reshape(-1) for a in cur_estimate]))))

            if inverse_hvp is None:
                inverse_hvp = np.array(cur_estimate) / ihvp_config['scale']
            else:
                inverse_hvp += np.array(cur_estimate) / ihvp_config['scale']

            # np.savetxt(self._path('debug_diffs_estimation_{}.txt'.format(sample_idx)), debug_diffs_estimation)

        inverse_hvp /= ihvp_config['num_repeats']
        return inverse_hvp

    def _estimation_op(self, v):
        v_const = [tf.constant(a, dtype=tf.float32, shape=a.shape) for a in v]
        damping = 1 - self.ihvp_config['damping']
        scale = self.ihvp_config['scale']
        return [
            a + (b * damping) - (c / scale)
            for a, b, c in zip(v_const, self.v_placeholder, self.hessian_vector_op)
        ]

    def _update_feed_dict(self, feed_dict, variables):
        for placeholder, variable in zip(self.v_placeholder, variables):
            feed_dict[placeholder] = variable
        return feed_dict

    def _grad_diffs(self, sess, train_indices, num_total_train_example):
        inverse_hvp = np.concatenate([a.reshape(-1) for a in self.inverse_hvp])
        grad_diff_op = self._grad_diff_op(num_total_train_example)

        num_to_remove = len(train_indices)
        predicted_grad_diffs = np.zeros([num_to_remove])

        for counter, idx_to_remove in enumerate(train_indices):
            single_data, single_label = self.feeder.train_one(idx_to_remove)
            feed_dict = self._make_feed_dict([single_data], [single_label])
            predicted_grad_diffs[counter] = self._grad_diff(sess, feed_dict, num_total_train_example, grad_diff_op,
                                                            inverse_hvp)

            if (counter % 1000) == 0:
                print('counter: {} / {}'.format(counter, num_to_remove))

        return predicted_grad_diffs

    def _grad_diffs_all(self, sess, train_batch_size, num_iters, num_subsampling):
        num_total_train_example = num_iters * train_batch_size
        if num_subsampling > 0:
            num_diffs = num_iters * num_subsampling
        else:
            num_diffs = num_iters * train_batch_size

        inverse_hvp = np.concatenate([a.reshape(-1) for a in self.inverse_hvp])
        grad_diff_op = self._grad_diff_op(num_total_train_example)
        predicted_grad_diffs = np.zeros([num_diffs])

        counter = 0
        for it in range(num_iters):
            train_batch_data, train_batch_label = self.feeder.train_batch(train_batch_size)

            if num_subsampling > 0:
                for idx in range(num_subsampling):
                    feed_dict = self._make_feed_dict(train_batch_data[idx:idx+1], train_batch_label[idx:idx+1])
                    predicted_grad_diffs[counter] = self._grad_diff(sess, feed_dict, num_total_train_example,
                                                                    grad_diff_op, inverse_hvp)
                    counter += 1
            else:
                for single_data, single_label in zip(train_batch_data, train_batch_label):
                    feed_dict = self._make_feed_dict([single_data], [single_label])
                    predicted_grad_diffs[counter] = self._grad_diff(sess, feed_dict, num_total_train_example,
                                                                    grad_diff_op, inverse_hvp)
                    counter += 1

            if (it % 100) == 0:
                print('iter: {}/{}'.format(it, num_iters))

        return predicted_grad_diffs

    def _grad_diff_op(self, num_train_example):
        inverse_hvp = np.concatenate([a.reshape(-1) for a in self.inverse_hvp])
        inverse_hvp_const = tf.reshape(tf.constant(inverse_hvp, dtype=tf.float64), shape=(-1, 1))
        flatten_grads = tf.concat([tf.reshape(a, (-1,)) for a in self.grad_op_train], 0)
        flatten_grads = tf.reshape(flatten_grads, shape=(1, -1,))
        flatten_grads /= num_train_example
        return tf.matmul(tf.cast(flatten_grads, tf.float64), inverse_hvp_const)

    def _grad_diff(self, sess, feed_dict, num_total_train_example, loss_diff_op, inverse_hvp):
        if _using_fully_tf:
            return sess.run(loss_diff_op, feed_dict=feed_dict)
        else:
            train_grads = sess.run(self.grad_op_train, feed_dict=feed_dict)
            train_grads = np.concatenate([a.reshape(-1) for a in train_grads])
            train_grads /= num_total_train_example
            return np.dot(inverse_hvp, train_grads)
