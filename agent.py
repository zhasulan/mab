import random

import numpy as np

from environment import Environment


class Agent(object):

    def __init__(self, environment):
        self.environment = environment
        pass

    def explore(self):
        return np.random.choice(self.environment.bandits)

    def exploit(self):
        return np.argmax(self.environment.theta)

    pass

