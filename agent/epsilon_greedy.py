import numpy as np

from agent.agent import Agent


class EpsilonGreedy(Agent):

    def __init__(self, actions, epsilon):
        super().__init__(actions)
        self.epsilon = epsilon
        pass

    def pull(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.actions)
        else:
            return np.argmax(self.Q)
