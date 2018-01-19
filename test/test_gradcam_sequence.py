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
from tensorflow.contrib import learn

class TestGradcamSequence(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        x_raw = ["a masterpiece of four years in the making"]
        vocab_path = "test/data/sequence/vocab"
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        self.x_test_batch = np.array(list(vocab_processor.transform(x_raw)))
        self.y_test_batch = [[1.0, 0.0]]
        
    def test_text(self):
        sess = tf.InteractiveSession()
        checkpoint_file = "test/data/sequence/model-15000"
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        graph = tf.get_default_graph()   
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]

        conv_op_names = darkon.Gradcam.candidate_featuremap_op_names(sess, 
            feed_options={input_x: self.x_test_batch, input_y: self.y_test_batch ,dropout_keep_prob:1.0})
                
        prob_op_names = darkon.Gradcam.candidate_predict_op_names(sess, 2, 
            feed_options={input_x: self.x_test_batch, input_y: self.y_test_batch ,dropout_keep_prob:1.0})
        
        conv_name = conv_op_names[-7]
        prob_name = prob_op_names[-1]
        self.assertEqual(conv_name, "conv-maxpool-3/relu")
        self.assertEqual(prob_name, "output/scores")
            
        insp = darkon.Gradcam(input_x, 2, conv_name, prob_name, graph=graph)
        ret = insp.gradcam(sess, self.x_test_batch[0], feed_options={dropout_keep_prob: 1})