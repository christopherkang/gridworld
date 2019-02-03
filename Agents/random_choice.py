from .agent import Agent
import numpy as np


class RandomAgent(Agent):
    def execute_policy(self):
        choice = self.ACTION_BANK[np.random.randint(0, 4)]
        return choice
