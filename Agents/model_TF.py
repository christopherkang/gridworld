import os

import tensorflow as tf

from agent import Agent


class Model_TF(Agent):
    def __init__(self, world, epsilon, decay, availableActions, dimensions,
                 learning_rate=0.2):
        super().__init__(self, world, epsilon, decay, availableActions, dimensions)
        self.LEARNING_RATE = learning_rate

    def consumeReward(self, reward, world):
        """Consume reward. Modifies original definition

        Arguments:
            Agent {Agent} -- agent
            reward {int} -- reward to be input
            world {Gridworld} -- world agent is in
        """

        self.totalReward += reward
        varRate = self.gradientDescent(self.LEARNING_RATE, reward,
                                       self.prediction_0, world.getRepresentation(True, True))
        print(
            f"Reward: {reward} | Total Reward: {self.totalReward} | varRate: {varRate} ")

    def save_model(self, directoryNum, roundNum, epoch):
        super.save_model(directoryNum, roundNum, epoch)
        self.saver.save(
            self.sess, f"/BasicGridworld/Models/tmp{directoryNum}/r{roundNum}/modele{epoch}.ckpt")

    def restore_model(self, directoryNum, roundNum, epoch):
        self.saver.restore(
            self.sess, f"/BasicGridworld/Models/tmp{directoryNum}/r{roundNum}/modele{epoch}.ckpt")
        print("Model restored")

    def end_round(self, directoryNum, roundNum):
        with open(f"/BasicGridworld/Models/tmp{directoryNum}/r{roundNum}/summary.txt", "w") as f:
            f.write(f"TOTAL_REWARD: {self.totalReward}")
        self.totalReward = 0

    def TFINIT(self):
        self.sess = tf.Session()  # config=tf.ConfigProto(log_device_placement=True)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def gradientDescent(self, learningRate, reward, prediction_0, environment):
        # gradients = self.optimizer.compute_gradients(self.model)
        # variationRate = learningRate * (
        #     reward + self.DECAY * self.predict_value(environment) - (prediction_0))
        # evaluatedGradients = [(tf.math.scalar_mul(np.asscalar(variationRate), gv[0]), gv[1])
        #                       for gv in gradients]

        # self.optimizer.apply_gradients(evaluatedGradients)
        currentPred = self.predict_value(environment)
        variationRate = learningRate * \
            (reward + self.DECAY * currentPred - (prediction_0))
        self.optimizer.minimize(self.model * variationRate)
        # create custom loss
        # ANN too
        # test linear in numpy w/ gradient
        # potential debug cnn w/ good dataset
        return variationRate
