import numpy as np

from environment import Environment


def get_agent(agent_name, actions, **agent_parameters):
    """

    :param agent_name:
    :param actions:
    :param agent_parameters:
    :return:
    """
    if agent_name == "greedy":
        from agent.greedy import Greedy
        return Greedy(actions)
        pass
    elif agent_name == "epsilon_greedy":
        from agent.epsilon_greedy import EpsilonGreedy
        return EpsilonGreedy(actions, agent_parameters["epsilon"])
    else:
        from agent.agent import Agent
        return Agent


class Bandit(object):

    def __init__(self, pulls, actions, agent_name, **agent_parameters):
        """

        :param pulls: Count of pulls
        :param actions: Count of actions
        :param agent_name: Name of Agent
        :param agent_parameters: Parameters of Agent
        """
        self.pulls = pulls
        self.agent = get_agent(agent_name, actions, **agent_parameters)
        self.env = Environment(pulls, actions)
        pass

    def experiment(self):
        """

        :rtype: (np.ndarray, np.ndarray)
        """
        rewards = np.zeros(self.pulls)
        optimal_actions = np.zeros(self.pulls)

        for pull in range(self.pulls):
            action = self.agent.pull()
            reward, optimal_action = self.env.get_reward(pull, action)
            self.agent.update_value(action, reward)

            rewards[pull] = reward
            optimal_actions[pull] = optimal_action
            pass

        return rewards, optimal_actions

    pass
