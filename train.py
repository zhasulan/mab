import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from bandit import Bandit


class Train(object):

    def __init__(self, experiments, pulls, actions):
        self.actions = actions
        self.pulls = pulls
        self.experiments = experiments

        self.values = {}
        pass

    def run(self, name, agent_type, **parameters):
        rewards = np.zeros(self.pulls)
        optimal_actions = np.zeros(self.pulls)

        for _ in tqdm(range(self.experiments)):
            bandit = Bandit(self.pulls, self.actions, agent_type, **parameters)
            reward, optimal_action = bandit.experiment()
            rewards += reward
            optimal_actions += optimal_action

            pass

        rewards /= np.float(self.experiments)
        optimal_actions /= np.float(self.experiments)

        self.values[name] = {}
        self.values[name]['rewards'] = rewards
        self.values[name]['optimal_actions'] = optimal_actions

        pass

    def print(self):

        plt.subplot(2, 1, 1)
        plt.plot([1.55 for _ in range(self.pulls)], linestyle="--", lw=1)

        legend = ['Best Possible']
        for key, value in self.values.items():
            legend.append(key)
            plt.plot(value['rewards'], '-', lw=1)

        plt.legend(legend)

        plt.title("Average Reward of Agents")
        plt.xlabel("Steps")
        plt.ylabel("Average reward")

        plt.ylim(0, 2.0)

        plt.subplot(2, 1, 2)
        plt.plot([100 for _ in range(self.pulls)], linestyle="--", lw=1)

        legend = ['Best Possible']
        for key, value in self.values.items():
            legend.append(key)
            plt.plot(value['optimal_actions'] * 100, '-', lw=1)

        plt.legend(legend)

        plt.title("Optimal Actions of Agents")
        plt.xlabel("Steps")
        plt.ylabel("% Optimal action")

        plt.ylim(0, 110)
        plt.show()

    pass
