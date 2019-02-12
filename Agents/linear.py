import numpy as np

from .model_NP import Model_NP
from .weight_generator import w_generator


class Linear(Model_NP):
    def __init__(self, epsilon, decay, availableActions,
                 dimensions, weight_scheme="RAND", learning_rate=0.001):
        super().__init__(self, epsilon, decay, availableActions, dimensions)
        self.WEIGHTS_SET = False
        self.LEARNING_RATE = learning_rate
        self.WEIGHT_SCHEME = weight_scheme
        # randomly initialize weights from 0 to 10

    def gradient_descent(self, rewards):
        """Apply the TD(0) gradient descent scheme

        Arguments:
            rewards {int} -- reward from the action just taken
        """

        var_rate = self.LEARNING_RATE * \
            (rewards + self.DECAY * self.predict_value() - self.PREDICTION_0)
        print(f"varRate: {var_rate}")
        print("")
        # self.weights += var_rate * self.distances
        delta = var_rate * np.sum(self.distances, axis=0)
        self.weights += delta

    def mc_update(self, episode):
        # matrix structured as Reward, state value, x_sum, y_sum

        for epoch in episode:
            delta = self.LEARNING_RATE * (epoch[0] - epoch[1])
            self.weights += delta * epoch[[2, 3]]
            self.bias += delta

    def init_weights(self):
        """Initializes weights

        Raises:
            Exception -- if the weight scheme given during initialization
            does not exist an appropriate scheme, throw an exception
        """

        w = w_generator._GENERATION_TYPES[self.WEIGHT_SCHEME]
        print(w[1])

        if (self.WEIGHT_SCHEME == "RAND"):
            # sample from random uniform distribution of (-1, 1)
            self.weights = w[0]((1, 2), -1, 1)
            self.bias = w[0]((1), -1, 1)
        elif (self.WEIGHT_SCHEME == "CONST"):
            # self.weights = np.zeros(((self.WORLD.item_list.shape)[0], 2))
            self.weights = w[0]((1, 2), 0)
            self.bias = w[0]((1), 0)
        else:
            raise Exception('Unknown Weight Scheme')

    def execute_policy(self):
        """Execute the optimal policy

        Returns:
            int -- returns the int corresponding to the action
            (index of the action in the matrix of self.ACTION_EFFECTS)
        """

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

    def predict_value(self, actionIndex="", debug=False, return_sums=False):
        """Produces state value prediction.

        Arguments:
            environment {Gridworld} -- gridworld object
            actionIndex {int} -- int of the action
        """

        if not actionIndex:
            proximity_map = self.WORLD.update_proximity_map(
                (0, 0), speculative=True)
        else:
            proximity_map = self.WORLD.update_proximity_map(
                self.ACTION_EFFECTS[actionIndex], speculative=True)

        self.distances = abs(proximity_map[:, 1:])
        available_items = proximity_map[:, 0] != 0

        # product_sums = np.sum(
        #     self.distances[available_items] *
        #     self.weights[available_items],
        #     axis=1)
        # value = np.sum(product_sums)

        sums = np.sum(self.distances[available_items], axis=0)

        value = np.sum(sums * self.weights) + self.bias

        if return_sums:
            return value, sums

        return value

    def load_model(self, directory):
        """Load a model from a specific directory.
        Should be saved as an np.savetxt file.

        Arguments:
            directory {str} -- directory of the filename.
        """

        self.weights = np.loadtxt(directory)

    def get_weights(self):
        """Helper for the model save method.
        Should return the relevant weights of a model that need to be saved.

        Returns:
            matrix -- matrix of the weights
        """
        return self.weights
