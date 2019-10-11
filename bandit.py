import numpy as np

from environment import Environment


class Bandit(object):

    def __init__(self, environment):
        self.environment = environment
        pass

    def reward(self):
        return np.random.uniform(0, 1, self.environment.num_bandits)

    pass


if __name__ == "__main__":
    env = Environment(10)
    bandit = Bandit(env)
    reward = bandit.reward()
    k = np.argmax(reward)
    print(reward)
    print(k, ': ', reward[k])
    pass
