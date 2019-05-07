from .agent import Agent
import numpy as np


class Model_NP(Agent):
    def save_model(self, directoryNum, roundNum, epoch):
        """Save the model

        Arguments:
            Agent {Agent} -- agent class
            directoryNum {int} -- directory num
            roundNum {int} -- round num
            epoch {int} -- epoch
        """

        super().save_model(directoryNum, roundNum, epoch)
        directory = f"{directoryNum}/r{roundNum}/e{epoch}"
        print(f"Saved agent info at {directory}")
        save_items = self.get_weights()
        for item_to_save in save_items.keys():
            np.savetxt(
                directory + item_to_save + ".txt",
                save_items[item_to_save])

    def get_weights(self):
        """Skeleton method. Return model weights. Every child should redefine this
        """

        pass
