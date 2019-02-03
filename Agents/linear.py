from .model_NP import Model_NP
import numpy as np


class Linear(Model_NP):
    def __init__(self, epsilon, decay, availableActions, dimensions, weight_scheme="RAND", learning_rate=0.001):
        super().__init__(self, epsilon, decay, availableActions, dimensions)
        self.WEIGHTS_SET = False
        self.LEARNING_RATE = learning_rate
        self.WEIGHT_SCHEME = weight_scheme
        # randomly initialize weights from 0 to 10

    def gradient_descent(self, rewards):
        # formula :
        # w <- w + alpha [R + gamma v(s') - v(s)]del v(s) wrt w
        # because this is a linear model, del is 1
        # w <- w + alpha [R + gamma v(s') - v(s)]
        var_rate = self.LEARNING_RATE * \
            (rewards + self.DECAY * self.predict_value() - self.PREDICTION_0)
        print(var_rate)
        self.weights -= var_rate * self.distances

    def init_weights(self):
        if (self.WEIGHT_SCHEME == "RAND"):
            self.weights = np.random.rand((self.WORLD.itemList.shape)[0], 2)
        elif (self.WEIGHT_SCHEME == "ZERO"):
            self.weights = np.zeros(((self.WORLD.itemList.shape)[0], 2))
        else:
            raise Exception('Unknown Weight Scheme')

    def execute_policy(self):
        self.PREDICTION_0 = self.predict_value()
        action = super().execute_policy()
        # self.PREDICTION_0 = self.predict_value(actionIndex=action)
        return action

    def set_world(self, world):
        """Redefine the internal pointer to the agent's environment

        Arguments:
            world {Gridworld} -- agent's current environment
        """

        self.WORLD = world
        if not self.WEIGHTS_SET:
            self.init_weights()
            self.WEIGHTS_SET = True
        # print(world)

    def predict_value(self, actionIndex="", debug=False):
        """Produces state value prediction.

        Arguments:
            environment {Gridworld} -- gridworld object
            actionIndex {int} -- int of the action
        """
        # 0 - wait let's actually square 1, 2 lol bc then it'll avoid negs HAH
        # 1. multiply 1,2 of proxmap with weights
        # 2. sum products
        # 3. mutliply 0 with products
        # 4. sum products

        # this is essentially gradient ASCENT

        def squarer(x): return x**2
        if not actionIndex:
            proximity_map = self.WORLD.update_proximity_map(
                (0, 0), speculative=True)
        else:
            proximity_map = self.WORLD.update_proximity_map(
                self.ACTION_EFFECTS[actionIndex], speculative=True)

        # distances = squarer(proximity_map[:, 1:])
        self.distances = abs(proximity_map[:, 1:])

        product_sums = np.sum(self.distances * self.weights, axis=1)

        value = -np.sum(product_sums)

        return value

    def load_model(self, directory):
        self.weights = np.loadtxt(directory)

    def get_weights(self):
        return self.weights
