from base.base_model import BaseModel
import tensorflow as tf


class MnistLogisticModel(BaseModel):
    def __init__(self, config):
        super(MnistLogisticModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, 784])  # mnist data image of shape 28*28=784
        self.y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes

        # Set model weights
        self.W = tf.Variable(tf.zeros([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))

        # Construct model
        self.pred = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)

        # Minimize using cross entropy
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(self.pred), reduction_indices=1))

        # Gradient descent
        self.optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.cost)

        # Evaluation
        correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

