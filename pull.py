import argparse

from train import Train


def main(actions, experiments, pulls):
    train = Train(experiments, pulls, actions)
    train.run("Greedy", "greedy")
    train.run("Epsilon greedy with epsilon=0.1", "epsilon_greedy", epsilon=0.1)
    train.run("Epsilon greedy with epsilon=0.01", "epsilon_greedy", epsilon=0.01)

    train.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi Armed Bandit testbed')
    parser.add_argument('--arms', required=True, help='Count of arms', default=10)
    parser.add_argument('--experiments', required=True, help='path to schema', default=2000)
    parser.add_argument('--pulls', required=True, help='path to dem', default=3000)
    args = parser.parse_args()

    main(int(args.arms), int(args.experiments), int(args.pulls))
    pass
