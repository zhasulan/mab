from agent import Agent
from bandit import Bandit
from environment import Environment

if __name__ == "__main__":
    env = Environment(10)
    bandit = Bandit(env)
    agent = Agent(env)

    # initialize
    env.theta = None
