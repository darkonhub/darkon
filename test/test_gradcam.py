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
from tensorflow.contrib.slim.nets import vgg
import tensorflow.contrib.slim as slim
import cv2


def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    resized_image = cv2.resize(image, (224, 224))
    return resized_image.astype(np.float)


def load_expected_data(model_name, meta):
    heatmap_path = 'test/data/gradcam/{}_gradcam_heatmap_{}.npy'.format(meta, model_name)
    guided_backprop_path = 'test/data/gradcam/{}_guided_backprop_{}.npy'.format(meta, model_name)
    gradcam_img_path = 'test/data/gradcam/{}_gradcam_img_{}.png'.format(meta, model_name)
    guided_gradcam_img_path = 'test/data/gradcam/{}_guided_gradcam_img_{}.png'.format(meta, model_name)
    return {
        'gradcam_img': load_image(gradcam_img_path),
        'guided_gradcam_img': load_image(guided_gradcam_img_path),
        'heatmap': np.load(heatmap_path),
        'guided_backprop': np.load(guided_backprop_path)
    }


def save_expected_data(model_name, meta, ret):
    heatmap_path = 'test/data/gradcam/{}_gradcam_heatmap_{}.npy'.format(meta, model_name)
    guided_backprop_path = 'test/data/gradcam/{}_guided_backprop_{}.npy'.format(meta, model_name)
    gradcam_img_path = 'test/data/gradcam/{}_gradcam_img_{}.png'.format(meta, model_name)
    guided_gradcam_img_path = 'test/data/gradcam/{}_guided_gradcam_img_{}.png'.format(meta, model_name)

    np.save(heatmap_path, ret['heatmap'])
    np.save(guided_backprop_path, ret['guided_backprop'])
    cv2.imwrite(gradcam_img_path, ret['gradcam_img'])
    cv2.imwrite(guided_gradcam_img_path, ret['guided_gradcam_img'])


class TestGradcam(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.nbclasses = 1000
        self.inputs = tf.placeholder(tf.float32, [1, 224, 224, 3])

    def tearDown(self):
        image = load_image('test/data/cat_dog.png')

        prob_op_name = darkon.Gradcam.candidate_predict_op_names(self.sess, self.nbclasses, self.graph_origin)[-1]
        insp = darkon.Gradcam(self.inputs, self.nbclasses, self.target_op_name, prob_op_name, graph=self.graph_origin)
        ret_top1 = insp.gradcam(self.sess, image)
        ret_243 = insp.gradcam(self.sess, image, 243)

        # save_expected_data(self.model_name, 'top1', ret_top1)
        # save_expected_data(self.model_name, '243', ret_243)

        exp_top1 = load_expected_data(self.model_name, 'top1')
        for key in ret_top1.keys():
            atol = 5 if 'img' in key else 1e-6
            print(key, atol)
            self.assertTrue(np.allclose(ret_top1[key], exp_top1[key], atol=atol))

        exp_243 = load_expected_data(self.model_name, '243')
        for key in ret_243.keys():
            atol = 5 if 'img' in key else 1e-6
            self.assertTrue(np.allclose(ret_243[key], exp_243[key], atol=atol))

        # just check new output
        # cv2.imwrite('test_{}.png', ret['gradcam_img'])
        # cv2.imwrite('test_guided_{}.png', ret['guided_gradcam_img'])
        # cv2.imwrite('test_guided_backprop_{}.png', ret['guided_backprop_img'])
        # tf.summary.FileWriter("./tmp/log-{}/".format(self.model_name), sess.graph)
        self.sess.close()

    def test_resnet(self):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = resnet_v1.resnet_v1_50(self.inputs, self.nbclasses, is_training=False)
        saver = tf.train.Saver(tf.global_variables())
        check_point = 'test/data/resnet_v1_50.ckpt'
        sess = tf.InteractiveSession()
        saver.restore(sess, check_point)

        self.sess = sess
        self.graph_origin = tf.get_default_graph()
        self.target_op_name = darkon.Gradcam.candidate_featuremap_op_names(sess, self.graph_origin)[-1]
        self.model_name = 'resnet'
        
        self.assertEqual('resnet_v1_50/block4/unit_3/bottleneck_v1/Relu', self.target_op_name) 

    def test_vgg(self):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            net, end_points = vgg.vgg_16(self.inputs, self.nbclasses, is_training=False)
            net = slim.softmax(net)
        saver = tf.train.Saver(tf.global_variables())
        check_point = 'test/data/vgg_16.ckpt'

        sess = tf.InteractiveSession()
        saver.restore(sess, check_point)

        self.sess = sess
        self.graph_origin = tf.get_default_graph()
        self.target_op_name = darkon.Gradcam.candidate_featuremap_op_names(sess, self.graph_origin)[-2]
        self.model_name = 'vgg'
        self.assertEqual('vgg_16/conv5/conv5_3/Relu', self.target_op_name)
