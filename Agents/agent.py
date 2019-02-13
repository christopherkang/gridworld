import os

import numpy as np
import tensorflow as tf


class Agent:
    def __init__(self, world, epsilon, decay, availableActions, dimensions):
        """Basic init for the Agent class.

        Arguments:
            world {pointer to Gridworld} -- pointer to Gridworld agent is in
            epsilon {int} -- epsilon value to use for e-greedy approach
            decay {int} -- gamma to use in gradient descent
            availableActions {[[], []]} -- nested list with action info
            dimensions {tuple} -- info with the world size
        """

        self.WORLD = world
        self.ACTION_BANK = availableActions[0]
        self.ACTION_EFFECTS = availableActions[1]
        self.total_reward = 0
        self.EPSILON = epsilon
        self.DECAY = decay

    def set_world(self, world):
        """Redefine the internal pointer to the agent's environment

        Arguments:
            world {Gridworld} -- agent's current environment
        """

        self.WORLD = world
        print(world)

    def execute_policy(self, print_action=True):
        """Execute policy

        Arguments:

        """
        max_index = []
        max_value = -float("inf")
        number_of_actions = len(self.ACTION_BANK)
        state_values = []
        probability_list = np.zeros((number_of_actions))

        for index in range(number_of_actions):  # for each action
            actionVal = np.asscalar((self.predict_value(index)))
            # predict the state value of the action
            state_values.append(actionVal)
            if actionVal > max_value:
                # if the value is greater than the previous
                max_index = [index]
                max_value = actionVal
            elif actionVal == max_value:
                max_index.append(index)

        print(state_values)
        is_experimental = np.random.choice(
            2, p=[1 - self.EPSILON, self.EPSILON]) == 1

        if (is_experimental):
            probability_list += 1 / number_of_actions
        else:
            probability_list[max_index] += 1.0 / len(max_index)

        choice = int(
            np.random.choice(
                number_of_actions,
                1,
                p=probability_list))
        if print_action:
            print(self.ACTION_BANK[choice])
        return choice

    def consume_reward(self, reward, round, world=""):
        """Consume reward

        Arguments:
            reward {int} -- reward value
            world {GridWorld} -- not necessary
        """

        self.total_reward += reward
        # self.gradient_descent(reward)
        print(
            f"Reward: {reward} | Total Reward: {self.total_reward} | Round: {round}")
        print()

    def save_model(self, directoryNum, roundNum, epoch):
        """Save model - only creates the directory if necessary.
        Child models should define which weights need to be saved, etc.

        Arguments:
            directoryNum {int} -- directory num
            roundNum {int} -- round num
            epoch {int} -- epoch
        """

        save_path = f"/Models/tmp{directoryNum}/r{roundNum}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def end_round(self, RUN_NUM=-1, round_num=-1):
        """End round and record necessary info. Reset total reward

        Keyword Arguments:
            RUN_NUM {int} -- the current run (default: {-1})
            round_num {int} -- the current round (default: {-1})
        """

        self.total_reward = 0
        pass

    def gradient_descent(self):
        """Empty method that almost all models will have.
        """

        pass
