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
"""
References
----------
.. [1] Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra \
"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" ICCV2017

"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import cv2
from skimage.transform import resize as skimage_resize

from .guided_grad import replace_grad_to_guided_grad
from .candidate_ops import candidate_featuremap_op_names, candidate_predict_op_names
from .candidate_ops import _unusable_ops


def _deprocess_image(x):
    # Same normalization as in:
    # https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


class Gradcam:
    """ Gradcam Class

    Parameters
    ----------
    x_placeholder : tf.Tensor
        Data place holder
        Tensor from tf.placeholder()
    num_classes: int
        number of classes
    featuremap_op_name : str
        Operation name of CNN feature map layer
        To get the list of candidate names, use ``Gradcam.candidate_featuremap_op_names()``
    predict_op_name : str
        Operation name of prediction layer (decision output)
        To get the list of candidate names, use ``Gradcam.candidate_predict_op_names()``
    graph : tf.Graph
        Tensorflow graph

    """
    def __init__(self, x_placeholder, num_classes, featuremap_op_name, predict_op_name=None, graph=None):
        self._x_placeholder = x_placeholder
        graph = graph if graph is not None else tf.get_default_graph()
        self.graph = graph

        predict_op_name = self._find_prob_layer(predict_op_name, graph)
        self._prob_ts = graph.get_operation_by_name(predict_op_name).outputs[0]
        self._target_ts = graph.get_operation_by_name(featuremap_op_name).outputs[0]

        self._class_idx = tf.placeholder(tf.int32)
        top1 = tf.argmax(tf.reshape(self._prob_ts, [-1]))

        loss_by_idx = tf.reduce_sum(tf.multiply(self._prob_ts, tf.one_hot(self._class_idx, num_classes)), axis=1)
        loss_by_top1 = tf.reduce_sum(tf.multiply(self._prob_ts, tf.one_hot(top1, num_classes)), axis=1)
        self._grad_by_idx = self._normalize(tf.gradients(loss_by_idx, self._target_ts)[0])
        self._grad_by_top1 = self._normalize(tf.gradients(loss_by_top1, self._target_ts)[0])

        replace_grad_to_guided_grad(graph)

        max_output = tf.reduce_max(self._target_ts, axis=2)
        self._saliency_map = tf.gradients(tf.reduce_sum(max_output), x_placeholder)[0]

    def gradcam(self, sess, input_data, target_index=None, feed_options=dict()):
        """ Calculate Grad-CAM (class activation map) and Guided Grad-CAM for given input on target class

        Parameters
        ----------
        sess: tf.Session
            Tensorflow session
        input_data : numpy.ndarray
            A single input instance
        target_index : int
            Target class index
            If None, predicted class index is used
        feed_options : dict
            Optional parameters to graph

        Returns
        -------
        dict

        Note
        ----
        Keys in return:
            * gradcam_img: Heatmap overlayed on input
            * guided_gradcam_img: Guided Grad-CAM result
            * heatmap: Heatmap of input on the target class
            * guided_backprop: Guided backprop result

        """
        input_feed = np.expand_dims(input_data, axis=0)
        if input_data.ndim == 3:
            is_image = True
            image_height, image_width = input_data.shape[:2]
        if input_data.ndim == 1:
            is_image = False
            input_length = input_data.shape[0]

        if target_index is not None:
            feed_dict = {self._x_placeholder: input_feed, self._class_idx: target_index}
            feed_dict.update(feed_options)
            conv_out_eval, grad_eval = sess.run([self._target_ts, self._grad_by_idx], feed_dict=feed_dict)
        else:
            feed_dict = {self._x_placeholder: input_feed}
            feed_dict.update(feed_options)
            conv_out_eval, grad_eval = sess.run([self._target_ts, self._grad_by_top1], feed_dict=feed_dict)

        weights = np.mean(grad_eval, axis=(0, 1, 2))
        conv_out_eval = np.squeeze(conv_out_eval, axis=0)
        cam = np.zeros(conv_out_eval.shape[:2], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * conv_out_eval[:, :, i]

        if is_image:
            cam += 1
            cam = cv2.resize(cam, (image_height, image_width))
            saliency_val = sess.run(self._saliency_map, feed_dict={self._x_placeholder: input_feed})
            saliency_val = np.squeeze(saliency_val, axis=0)
        else:
            cam = skimage_resize(cam, (input_length, 1), preserve_range=True, mode='reflect')
            cam = np.transpose(cam)

        cam = np.maximum(cam, 0)
        heatmap = cam / np.max(cam)

        ret = {'heatmap': heatmap}

        if is_image:
            ret.update({
                'gradcam_img': self.overlay_gradcam(input_data, heatmap),
                'guided_gradcam_img': _deprocess_image(saliency_val * heatmap[..., None]),
                'guided_backprop': saliency_val
            })
        return ret

    @staticmethod
    def candidate_featuremap_op_names(sess, graph=None, feed_options=None):
        """ Returns the list of candidates for operation names of CNN feature map layer

        Parameters
        ----------
        sess: tf.Session
            Tensorflow session
        graph: tf.Graph
            Tensorflow graph
        feed_options: dict
            Optional parameters to graph
        Returns
        -------
        list
            String list of candidates

        """
        graph = graph if graph is not None else tf.get_default_graph()
        feed_options = feed_options if feed_options is not None else {}
        return candidate_featuremap_op_names(sess, graph, feed_options)

    @staticmethod
    def candidate_predict_op_names(sess, num_classes, graph=None, feed_options=None):
        """ Returns the list of candidate for operation names of prediction layer

        Parameters
        ----------
        sess: tf.Session
            Tensorflow session
        num_classes: int
            Number of prediction classes
        graph: tf.Graph
            Tensorflow graph
        feed_options: dict
            Optional parameters to graph
        Returns
        -------
        list
            String list of candidates

        """
        graph = graph if graph is not None else tf.get_default_graph()
        feed_options = feed_options if feed_options is not None else {}
        return candidate_predict_op_names(sess, num_classes, graph, feed_options)

    @staticmethod
    def overlay_gradcam(image, heatmap):
        """ Overlay heatmap on input data
        """
        output_image = np.array(image)
        output_image -= np.min(output_image)
        output_image = np.minimum(output_image, 255)

        cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
        output_image = np.float32(cam) + np.float32(output_image)
        output_image = 255 * output_image / np.max(output_image)
        output_image = np.uint8(output_image)
        return output_image

    @staticmethod
    def _find_prob_layer(output_name, graph):
        if output_name is not None:
            return output_name

        for op in graph.get_operations():
            if _unusable_ops(op):
                continue

            output_name = op.name
        return output_name

    @staticmethod
    def _normalize(x):
        return tf.div(x, (tf.sqrt(tf.reduce_mean(tf.square(x), axis=1)) + 1e-5))
