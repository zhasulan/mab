import numpy as np


class Agent(object):

    last_action = 0

    def __init__(self):
        self.arm_use = np.zeros()
        self.q_values = 0
        pass

    def q_update(self):
        pass