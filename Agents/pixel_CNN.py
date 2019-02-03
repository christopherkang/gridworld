from .model_TF import Model_TF
import tensorflow as tf


class pixelCNNAgent(Model_TF):
    def __init__(self, epsilon, decay, availableActions, dimensions, learning_rate=0.2):
        super.super.__init__(
            epsilon, decay, availableActions, dimensions, learning_rate)

        xWidth = dimensions[0]
        yWidth = dimensions[1]
        self.pixelMap = tf.placeholder(
            tf.float16, [xWidth, yWidth]
        )
        self.pixelMap1 = tf.expand_dims(self.pixelMap, 2)
        self.pixelMap2 = tf.expand_dims(self.pixelMap1, 0)
        self.conv1 = tf.layers.conv2d(
            inputs=self.pixelMap2,
            filters=4,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu
        )
        self.conv2 = tf.layers.conv2d(
            inputs=self.conv1,
            filters=8,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
        )
        self.flat = tf.reshape(
            self.conv2,
            [-1, (xWidth) * (yWidth) * 8]
        )
        self.dense = tf.layers.dense(
            inputs=self.flat, units=128, activation=tf.nn.relu
        )
        self.model = tf.layers.dense(
            inputs=self.dense, units=1
        )
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.LEARNING_RATE)

    def predict_value(self, environment, actionIndex):
        """Predict the value of each move given the environment

        Arguments:
            environment {differential} -- environment representation
            depends on the chosen model for its type

        Returns:
            1 value -- value prediction of specified state
        """
        pixels = environment.simulateAction(
            self.ACTION_EFFECTS[actionIndex][0],
            self.ACTION_EFFECTS[actionIndex][1])
        prediction = self.sess.run(self.model, feed_dict={
                                   self.pixelMap: pixels})
        return prediction
