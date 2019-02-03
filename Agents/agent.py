import os

import numpy as np
import tensorflow as tf


class Agent:
    def __init__(self, world, epsilon, decay, availableActions, dimensions):
        self.WORLD = world
        self.ACTION_BANK = availableActions[0]
        self.ACTION_EFFECTS = availableActions[1]
        self.totalReward = 0
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
        maxIndex = -1
        maxValue = -np.nan
        numberOfActions = len(self.ACTION_BANK)
        stateVals = []

        for index in range(numberOfActions):  # for each action
            actionVal = np.asscalar((self.predict_value(index)))
            # predict the state value of the action
            stateVals.append(actionVal)
            if actionVal > maxValue:
                # if the value is greater than the previous
                maxIndex = index
                maxValue = actionVal
        print(stateVals)

        probMatrix = np.zeros((numberOfActions))
        if (maxIndex == -1):
            probMatrix += 1 / numberOfActions
        else:
            probMatrix += self.EPSILON / numberOfActions
            probMatrix[maxIndex] = 1 - self.EPSILON + \
                self.EPSILON / numberOfActions
        choice = int(np.random.choice(numberOfActions, 1, p=probMatrix))
        if print_action:
            print(self.ACTION_BANK[choice])
        return choice

    def consume_reward(self, reward, world=""):
        """Consume reward 

        Arguments:
            reward {int} -- reward value
            world {GridWorld} -- not necessary
        """

        self.totalReward += reward
        self.gradient_descent(reward)
        print(f"Reward: {reward} | Total Reward: {self.totalReward} ")

    def save_model(self, directoryNum, roundNum, epoch):
        save_path = f"./Models/tmp{directoryNum}/r{roundNum}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def end_round(self, run_NUM=-1, round_num=-1):
        self.totalReward = 0
        pass

    def gradient_descent(self):
        pass
