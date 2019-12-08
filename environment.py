import numpy as np


class Environment(object):

    def __init__(self, pulls, actions):
        self.pulls = pulls
        self.actions = actions
        self.arms = np.random.normal(0, 1, actions)
        self.q_star = np.random.normal(self.arms, 1, (pulls, actions))
        self.q_optimal = np.argmax(self.arms)

        pass

    def get_reward(self, pull, action):
        optimal = 0
        if self.q_optimal == action:
            optimal = 1
        return self.q_star[pull, action], optimal
