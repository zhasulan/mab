import numpy as np


class Agent(object):

    def __init__(self, actions):
        self.actions = actions
        self.Q = np.zeros(actions)
        self.action_count = np.zeros(actions)
        pass

    def update_value(self, action, reward):
        """

        :param action: Count of actions
        :param reward: Update Q value from reward
        """
        self.action_count[action] = self.action_count[action] + 1
        self.Q[action] += (1. / self.action_count[action]) * (reward - self.Q[action])
        pass

    def pull(self):
        return np.argmax(self.Q)
