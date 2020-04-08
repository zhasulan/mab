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
        """

        :param pull: Number of pull
        :param action: Number of action
        :return:
        """
        optimal = 0
        if self.q_optimal == action:
            optimal = 1
        return self.q_star[pull, action], optimal


if __name__ == "__main__":
    # np.random.seed(1)
    z = np.zeros(1000)

    for i in range(1000):
        env = Environment(1, 10)
        # print(env.arms)
        # print(env.q_optimal)
        # z[i] = np.mean(np.random.normal(env.arms[env.q_optimal], 1, 3000))
        z[i] = env.arms[env.q_optimal]
        z[i] = np.amax(env.arms)
    print(np.mean(z))
