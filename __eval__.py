

from tictactoe import TicTacToe, TicTacToePolicy
from pit import pit

import torch


def main():
    game = TicTacToe()
    policy = TicTacToePolicy()
    policy.load_state_dict(torch.load("saved/final"))

    print(pit(policy, policy, game, render=True))



if __name__ == "__main__":
    main()

