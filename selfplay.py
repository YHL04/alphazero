

import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import deque


from mcts import MCTS
from pit import pit


class SelfPlay:
    """
    Main class for training AlphaZero via self play.

    """

    def __init__(self, game, policy, opt, **kwargs):
        self.game = game
        self.prev_policy = deepcopy(policy)
        self.policy = policy
        self.opt = opt

        self.mcts = MCTS(self.game, self.policy, *kwargs)

    def run_episode(self, num_mcts, temp):
        """
        Runs one episode of self play
        """
        boards, pis = [], []
        board = self.game.get_init_board()
        player = 1

        z = None
        while z is None:
            pi = self.mcts.get_probs(board, player, num_mcts=num_mcts, temp=temp)
            boards.append(self.game.get_player_pov(board, player))
            pis.append(pi)

            action = np.random.choice(len(pi), p=pi)
            board, player = self.game.get_next_state(board, player, action)

            z = self.game.get_game_ended(board, 1)

        zs = [z if i % 2 == 0 else -z for i in range(len(pis))]
        # print("----------")
        # for b in boards:
        #     self.game.render(b)
        # print(zs)
        return np.array(boards), np.array(pis), np.array(zs)

    def train(self, iter, num_eps, num_mcts, buffer_size, temp):
        loss_v, loss_pi, last_pit = 0, 0, 0

        for it in range(iter):

            boards = deque([], maxlen=buffer_size)
            pis = deque([], maxlen=buffer_size)
            zs = deque([], maxlen=buffer_size)

            for _ in tqdm(range(num_eps), desc="Self Play, Last Loss: {:.6f}, {:.6f}, Last Pit: {}"
                    .format(loss_v, loss_pi, last_pit)):
                self.mcts = MCTS(self.game, self.policy)
                board, pi, z = self.run_episode(num_mcts, temp)
                boards.append(board)
                pis.append(pi)
                zs.append(z)

            # update network
            for _ in range(5):
                loss_v, loss_pi = self.policy.update(
                    x=torch.tensor(np.concatenate(boards, axis=0), dtype=torch.float32),
                    z=torch.tensor(np.concatenate(zs, axis=0), dtype=torch.float32).view(-1, 1),
                    pi=torch.tensor(np.concatenate(pis, axis=0), dtype=torch.float32),
                    opt=self.opt
                )

            # save model
            torch.save(self.policy.state_dict(), "saved/final")

            # evaluate
            last_pit = pit(self.policy, self.prev_policy, self.game)

            # cache this policy
            self.prev_policy.load_state_dict(self.policy.state_dict())

            p, v = self.policy(
                torch.tensor([[1., 0, 0],
                              [0, 0, -1],
                              [1, 0, -1]])
            )
            print(p)


