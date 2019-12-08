import numpy as np

from agent.agent import Agent


class Greedy(Agent):

    def __init__(self, actions):
        super().__init__(actions)
        pass

    def pull(self):
        return np.argmax(self.Q)
