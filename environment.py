import numpy as np


class Environment(object):

    def __init__(self, num_bandits, arm_count):
        self.num_bandits = num_bandits
        self.arm_count = arm_count
        self.q = np.random.normal(0, 1, (num_bandits, arm_count))
        self.q_optimal = np.argmax(self.q, 1)

        pass

    def pull(self, i, j):
        return np.random.normal(self.q[i][j], 1)


if __name__ == "__main__":
    env = Environment(3, 3)
    print(env.q)
