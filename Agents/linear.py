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

    # def save_model(self, directoryNum, roundNum, epoch):
        # super().save_model(directoryNum, roundNum, epoch)
        # np.savetxt(
        #     f"{directoryNum}/r{roundNum}/e{epoch}_weights.txt",
        #     self.weights)

        # np.savetxt(
        #     f"{directoryNum}/r{roundNum}/e{epoch}_bias.txt",
        #     self.bias)

        # MODEL_NP REDEFINES THIS METHOD
        # pass

    def gradient_descent(self, rewards):
        """Apply the TD(0) gradient descent scheme

        Arguments:
            rewards {int} -- reward from the action just taken
        """

        # FLAG THIS CHANGES DEPENDING UPON THE SCHEME

        var_rate = self.LEARNING_RATE * \
            (rewards + self.DECAY * self.predict_value() - self.PREDICTION_0)
        print(f"varRate: {var_rate}")
        # self.weights += var_rate * self.distances
        # abs(x) + abs(y)
        delta = var_rate * np.sum(self.distances, axis=0)
        self.weights += delta

    def mc_update(self, episode, partials):
        # matrix structured as Reward, state value, x_distances, y_distances

        delta_list = []

        for epoch in range(len(partials)):
            delta = self.LEARNING_RATE * \
                (episode[epoch, 0] - episode[epoch, 1])
            delta_list.append(delta)
            self.weights += float(delta) * abs(np.asarray(partials[epoch]))
            self.bias += delta

        self.bias = np.clip(self.bias, -1, 1)
        self.weights = np.clip(self.weights, -1, 1)
        # print(f"deltas:{delta_list}")

    def init_weights(self):
        """Initializes weights

        Raises:
            Exception -- if the weight scheme given during initialization
            does not exist an appropriate scheme, throw an exception
        """

        w = w_generator._GENERATION_TYPES[self.WEIGHT_SCHEME]
        print(w[1])

        # this needs to change given the number of weights
        weight_dims = (2, len(self.WORLD.item_list))
        if (self.WEIGHT_SCHEME == "RAND"):
            # sample from random uniform distribution of (-1, 1)
            self.weights = w[0](weight_dims, -1, 1)
            self.bias = w[0]((1), -1, 1)

        elif (self.WEIGHT_SCHEME == "CONST"):
            # self.weights = np.zeros(((self.WORLD.item_list.shape)[0], 2))
            self.weights = w[0](weight_dims, 0)
            self.bias = w[0]((1), 0)

        elif (self.WEIGHT_SCHEME == "IDEAL"):
            # THIS IS DEPENDENT UPON THE INPUTS IN THE ITEM LIST
            self.weights = w[0](weight_dims, 0) - 1
            self.weights[0] = [-10.0, 0.5, 0.5]
            self.weights[1] = [-10.0, 0.5, 0.5]
            self.bias = w[0]((1), 0.0)

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

    # def predict_value(self, actionIndex="", debug=False, return_sums=False):
    #     """Produces state value prediction.

    #     Arguments:
    #         environment {Gridworld} -- gridworld object
    #         actionIndex {int} -- int of the action
    #     """

    #     if actionIndex == "":  # if the action selected is the current state
    #         proximity_map = self.WORLD.update_proximity_map(
    #             (0, 0), speculative=True)
    #     else:  # refer to a specific action
    #         proximity_map = self.WORLD.update_proximity_map(
    #             self.ACTION_EFFECTS[actionIndex], speculative=True)

    #     self.distances = abs(proximity_map[:, 1:])
    #     # indices of all available objects
    #     available_items = np.where(proximity_map[:, 0] != 0)

    #     if available_items[0].shape[0] == 0:
    #         if return_sums:
    #             return 0, 0
    #         return self.bias  # RETURN BIAS

    #     # product_sums = np.sum(
    #     #     self.distances[available_items] *
    #     #     self.weights[available_items],
    #     #     axis=1)
    #     # value = np.sum(product_sums)

    #     sums = np.sum(self.distances[available_items], axis=0)

    #     value = np.sum(sums * self.weights) + self.bias

    #     if return_sums:
    #         # correct for the number of objects
    #         return value, sums / available_items[0].shape[0]

    #     # FLAG - PRODUCING ARRAYS AND INTS

    #     return value

    def predict_value(self, actionIndex="", debug=False, return_sums=False):
        """Produces state value prediction.

        Arguments:
            environment {Gridworld} -- gridworld object
            actionIndex {int} -- int of the action
        """

        if actionIndex == "":  # if the action selected is the current state
            proximity_map_x, proximity_map_y = self.WORLD.calculate_grid_map(
                (0, 0))
        else:  # refer to a specific action
            proximity_map_x, proximity_map_y = self.WORLD.calculate_grid_map(
                self.ACTION_EFFECTS[actionIndex])

        run_sum = self.bias[0]

        # FLAG THIS DISTANCE MAP IS JUST THE NUMBER OF MINIMUM MOVES NECESSARY,
        # NOT LINEAR DISTANCE
        # self.distances = abs(
        # proximity_map_x[1:, 0]) + abs(proximity_map_y[1:, 0])

        self.distances = proximity_map_x[1:, 0] + proximity_map_y[1:, 0]

        # self.distances isn't actually used in anything???

        for target in range(len(self.weights[0])):
            run_sum += abs(proximity_map_x[target + 1]
                           [0]) * self.weights[0][target]
            run_sum += abs(proximity_map_y[target + 1]
                           [0]) * self.weights[1][target]

        # indices of all available objects
        # available_items = np.where(proximity_map[:, 0] != 0)

        # if available_items[0].shape[0] == 0:
        #     if return_sums:
        #         return 0, 0
        #     return self.bias  # RETURN BIAS

        # product_sums = np.sum(
        #     self.distances[available_items] *
        #     self.weights[available_items],
        #     axis=1)
        # value = np.sum(product_sums)

        # sums = np.sum(self.distances[available_items], axis=0)

        # value = np.sum(sums * self.weights) + self.bias

        if return_sums:
            # correct for the number of objects
            # flag need to fix 0.0 - should be the
            return run_sum, [proximity_map_x[1:, 0].astype(
                float), proximity_map_y[1:, 0].astype(float)]
            # return value, sums / available_items[0].shape[0]

        # FLAG - PRODUCING ARRAYS AND INTS

        return run_sum

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
        return {'weights': self.weights, 'bias': self.bias}
