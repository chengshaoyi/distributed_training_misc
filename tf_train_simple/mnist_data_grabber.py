from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf


class DataGrab():
    def __init__(self, data_dir):
        self.mnist = input_data.read_data_sets(data_dir, one_hot=True)

    def get_next_train(self, batch_size):
        batch = self.mnist.train.next_batch(batch_size=batch_size)
        return batch

    def get_test(self):
        return self.mnist.test
