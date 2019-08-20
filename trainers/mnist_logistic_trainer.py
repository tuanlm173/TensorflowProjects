import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from base.base_train import BaseTrain
import numpy as np

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


class MnistLogisticTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(MnistLogisticTrainer, self).__init__(sess, model, data, config, logger)
        self.init = tf.global_variables_initializer()

    def train(self):
        self.sess.run(self.init)
        self.train_epoch()

    def train_epoch(self):
        loop = range(self.config.training_epochs)
        for epoch in loop:
            self.avg_cost = 0.
            total_batch = int(mnist.train.num_examples / self.config.batch_size)
            for i in range(total_batch):
                loss, acc = self.train_step()
                self.avg_cost += loss / total_batch
            if (epoch + 1) % self.config.display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(self.avg_cost))
        print("Optimization Finished!")
        print("Accuracy: ", acc * 100, "%")

    def train_step(self):
        batch_xs, batch_ys = mnist.train.next_batch(self.config.batch_size)
        _, c, acc = self.sess.run([self.model.optimizer, self.model.cost, self.model.accuracy],
                                  feed_dict={self.model.x: batch_xs, self.model.y: batch_ys})
        return c, acc
