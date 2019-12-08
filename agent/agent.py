import numpy as np


class Agent(object):

    def __init__(self, actions):
        self.actions = actions
        self.Q = np.zeros(actions)
        self.action_count = np.zeros(actions)
        # self.action_rewards = np.zeros(actions)
        pass

    def learn(self, action, reward):
        self.action_count[action] = self.action_count[action] + 1
        # self.action_rewards[action] = self.action_rewards[action] + reward
        # self.Q[action] = self.action_rewards[action] / self.action_count[action]
        self.Q[action] += (1. / self.action_count[action]) * (reward - self.Q[action])
        pass

    def pull(self):
        return np.argmax(self.Q)
