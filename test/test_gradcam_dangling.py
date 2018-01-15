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
from tensorflow.contrib.slim.nets import resnet_v1
import tensorflow.contrib.slim as slim
import cv2


def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    resized_image = cv2.resize(image, (224, 224))
    return resized_image.astype(np.float)


class TestGradcamDangling(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

        self.nbclasses = 1000
        inputs = tf.placeholder(tf.float32, [1, 224, 224, 3])
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = resnet_v1.resnet_v1_50(inputs, self.nbclasses, is_training=False)
        saver = tf.train.Saver(tf.global_variables())
        check_point = 'test/data/resnet_v1_50.ckpt'

        sess = tf.InteractiveSession()
        saver.restore(sess, check_point)

        conv_name = 'resnet_v1_50/block4/unit_3/bottleneck_v1/Relu'
        
        self.graph_origin = tf.get_default_graph().as_graph_def()
        self.insp = darkon.Gradcam(inputs, self.nbclasses, conv_name)
        self.sess = sess

    def tearDown(self):
        self.sess.close()

    def test_dangling(self):
        image = load_image('test/data/cat_dog.png')

        graph_influence_init = tf.get_default_graph().as_graph_def()
        self.assertNotEqual(self.graph_origin, graph_influence_init)

        _ = self.insp.gradcam(self.sess, image)

        graph_first_executed = tf.get_default_graph().as_graph_def()
        self.assertEqual(graph_influence_init, graph_first_executed)

        _ = self.insp.gradcam(self.sess, image)

        graph_second_executed = tf.get_default_graph().as_graph_def()
        self.assertEqual(graph_first_executed, graph_second_executed)

