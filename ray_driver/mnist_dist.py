import ray
import ray.tune as tune
from ray.tune.util import pin_in_object_store, get_pinned_object
import tensorflow as tf
from tf_train_simple.mnist_data_grabber import DataGrab
from tf_train_simple.mnist_model_builder import build_model

ray.init()
data = pin_in_object_store(DataGrab('/tmp/ray/tf/mnist/input_data'))


def train_func(config, reporter):  # add a reporter arg
    my_lr = config["lr"]
    cur_data = get_pinned_object(data)
    with tf.Session() as sess:
        x, y_, keep_prob, train_step, accuracy = build_model(my_lr)
        print('number of global vars:',len(tf.global_variables()))

        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            batch = cur_data.get_next_train(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, learning rate %f training accuracy %g' % (i, my_lr, train_accuracy))
                reporter(timesteps_total=i, mean_accuracy=train_accuracy)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        #test_data = data.get_test()
        #print('test accuracy %g' % accuracy.eval(feed_dict={
        #    x: test_data.images, y_: test_data.labels, keep_prob: 1.0}))

tune.register_trainable("train_func", train_func)

all_trials = tune.run_experiments({
    "my_experiment": {
        "run": "train_func",
        "stop": {"mean_accuracy": 0.5},
        "config": {
            "lr": tune.grid_search([0.001, 0.0006]),
        }
    }
})

