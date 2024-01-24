

from tictactoe import TicTacToe, TicTacToePolicy
from selfplay import SelfPlay

import torch.optim as optim


"""
Hyperparameters:

One gradient descent step per training iteration
Training episode terminates when game ends (MCTS cache has reached game end, reset after)

num_mcts = 25          (number of Monte Carlo Tree Search per training episode step)
num_eps = 100          (number of training episodes per training iteration)
cpuct = 1              (constant weight for value and upper confidence bound)
buffer_size = 100_000
lr = 1e-3
temp = 1


"""


def main(iter=200, num_eps=100, num_mcts=200, buffer_size=100_000, lr=1e-3, temp=0.8):

    game = TicTacToe()
    policy = TicTacToePolicy()
    opt = optim.SGD(policy.parameters(), lr=lr)

    selfplay = SelfPlay(game, policy, opt)
    selfplay.train(iter, num_eps, num_mcts, buffer_size, temp)


if __name__ == "__main__":
    main()

