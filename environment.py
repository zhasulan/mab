import numpy as np


class Environment(object):

    def __init__(self, num_bandits):
        self.num_bandits = num_bandits
        self.theta = np.zeros(num_bandits)

        pass


if __name__ == "__main__":
    env = Environment(10)
    print(env.theta)
