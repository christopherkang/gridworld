from .agent import Agent
import numpy as np


class Model_NP(Agent):
    def save_model(self, directoryNum, roundNum, epoch):
        super().save_model(directoryNum, roundNum, epoch)
        directory = f"/Models/tmp{directoryNum}/r{roundNum}/modele{epoch}.txt"
        np.savetxt(directory, self.get_weights())

    def get_weights(self):
        """Skeleton method. Return model weights. Every child should redefine this
        """

        pass
