import numpy as np

from environment import Environment


def get_agent(agent_type, actions, **parameters):
    if agent_type == "greedy":
        from agent.greedy import Greedy
        return Greedy(actions)
        pass
    elif agent_type == "epsilon_greedy":
        from agent.epsilon_greedy import EpsilonGreedy
        return EpsilonGreedy(actions, parameters["epsilon"])
    else:
        from agent.agent import Agent
        return Agent


class Bandit(object):

    def __init__(self, pulls, actions, agent_type, **agent_parameters):
        self.pulls = pulls
        self.agent = get_agent(agent_type, actions, **agent_parameters)
        self.env = Environment(pulls, actions)
        pass

    def experiment(self):
        rewards = np.zeros(self.pulls)
        optimal_actions = np.zeros(self.pulls)

        for pull in range(self.pulls):
            action = self.agent.pull()
            reward, optimal_action = self.env.get_reward(pull, action)
            self.agent.learn(action, reward)

            rewards[pull] = reward
            optimal_actions[pull] = optimal_action
            pass

        return rewards, optimal_actions

    pass
