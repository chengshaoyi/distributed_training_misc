# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

from tf_train_simple.mnist_data_grabber import DataGrab
from tf_train_simple.mnist_model_builder import build_model
import numpy as np
FLAGS = None


def train_func(config, reporter):  # add a reporter arg
    print("start data downloading")
    my_lr = config["lr"]
    data = DataGrab('/tmp/ray/tf/mnist/input_data' + str(my_lr))
    x, y_, keep_prob, train_step, accuracy, w1_conv1_mask = build_model(my_lr)
    print(w1_conv1_mask.get_shape().as_list())
    conv1_mask_one = np.ones(w1_conv1_mask.get_shape().as_list())
    conv1_mask_one[:,:,:,:8] = 0
    print(conv1_mask_one)
    print("starting tf session")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("done initialization")
        for i in range(2000):
            batch = data.get_next_train(64)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0, w1_conv1_mask: conv1_mask_one})
                print('step %d, training accuracy %g' % (i, train_accuracy))
                reporter(timesteps_total=i, mean_accuracy=train_accuracy)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, w1_conv1_mask: conv1_mask_one})
        test_data = data.get_test()
        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: test_data.images, y_: test_data.labels, keep_prob: 1.0, w1_conv1_mask: conv1_mask_one}))
        return tf.get_default_graph()

def main(_):
    config = {'lr':0.0002}
    reporter = lambda timesteps_total,mean_accuracy: print(timesteps_total, mean_accuracy)
    trained_graph = train_func(config, reporter)
    # after initial training, we should have the model
    op_to_trim = []
    '''for op in trained_graph.get_operations():
        if op.op_def.name == 'Conv2D':
            print('--------------',op.name)
            print(op)
            print('---------------')'''



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)