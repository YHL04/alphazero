

import torch
import torch.nn as nn
import torch.nn.functional as F


class TicTacToePolicy(nn.Module):

    def __init__(self):
        super(TicTacToePolicy, self).__init__()

        self.input = nn.Linear(9, 64)
        self.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(3)])

        self.pi = nn.Linear(64, 9)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.input(x.view(-1, 9)))
        for layer in self.layers:
            x = F.relu(layer(x))

        p = F.softmax(self.pi(x), dim=-1)
        v = F.tanh(self.value(x))

        return p, v

    @torch.no_grad()
    def play(self, x):
        """x is numpy.ndarray"""
        p, v = self.forward(torch.tensor(x, dtype=torch.float32).view(1, 3, 3))
        return p.squeeze().numpy(), v.squeeze().numpy()

    def update(self, x, z, pi, opt):
        """gradient descent step in self play, we get log probability of P but not pi"""
        B = x.size(0)
        assert x.shape == (B, 3, 3)
        assert z.shape == (B, 1)
        assert pi.shape == (B, 9)

        p, v = self.forward(x)

        loss_v = F.mse_loss(v, z)
        loss_pi = -torch.mean(torch.sum(pi * torch.log(p), dim=1))
        loss = loss_v + loss_pi
        loss.backward()

        opt.step()
        opt.zero_grad()

        return loss_v.detach().numpy(), loss_pi.detach().numpy()

