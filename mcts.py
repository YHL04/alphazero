

import numpy as np
import math


class MCTS:

    def __init__(self, game, policy):
        self.game = game
        self.policy = policy

        self.Qsa = {}  # stores Q values for s,a (averaged)
        self.Nsa = {}  # stores # of times s,a was visited
        self.Ns = {}   # stores # of times board s was visited
        self.Ps = {}   # stores initial policy

        self.Es = {}   # stores if game ended for board s
        self.Vs = {}   # stores valid moves for board s

    def get_probs(self, current, player, temp, num_mcts):
        """
        Performs MCTS simulations starting from current
        board and returns probability according to visited
        counts (which is determined by value predicted by policy
        and number of times its visited)

        TODO: still need to implement logic for temp = 0
        """
        for _ in range(num_mcts):
            self.search(current, player)

        s = self.game.get_rep(current)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
                  for a in range(self.game.get_action_size())]

        counts = [x ** (1./temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, current, player, cpuct=1., eps=1e-8):
        """
        Performs one iteration of MCTS.

        First checks if game ended, cache it.
        If no cached policy, then generate policy and value, then cache it and return.
        Then, get action according to # of visits, policy cache, and value cache
        The search ends when game ends, or when no cached policy is found,
        otherwise it does recursion.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player
        after the current player has made a move.

        Parameters:
            current (Board): current board class representation
            cpuct (float): constant to be multiplied by P
            eps (float): offset if # visited is 0
        """
        s = self.game.get_rep(current)

        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(current, player)
        if self.Es[s] is not None:
            # terminal node (either -1, 0, 1)
            return -self.Es[s]

        # if can't find model policy, get it from model, cache then return
        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.policy.play(self.game.get_player_pov(current, player))

            # mask out invalid moves and re-normalize
            valids = self.game.get_valid_moves(current)
            self.Ps[s] = self.Ps[s] * valids
            sum_ps_s = np.sum(self.Ps[s])

            if sum_ps_s > 0:
                self.Ps[s] /= sum_ps_s
            else:
                raise ValueError("sum_ps_s not > 0")

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float("inf")
        best_act = -1

        for a in range(self.game.get_action_size()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + \
                        cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / \
                        (1 + self.Nsa[(s, a)])

                else:
                    u = cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + eps)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.get_next_state(current, 1, a)

        v = self.search(next_s, next_player)

        # if Q already exists, add to average, else create new Q
        # Q is basically the value of the next state after taking action a
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
