

import torch


@torch.no_grad()
def pit(policy1, policy2, game, render=False):
    """returns whether policy1 won against policy2, draw, or lost"""
    board = game.get_init_board()
    player = 1

    done = None
    while done is None:
        board_in = game.get_player_pov(board, player)

        if player == 1:
            p, v = policy1(torch.tensor(board_in, dtype=torch.float32))
        else:
            p, v = policy2(torch.tensor(board_in, dtype=torch.float32))

        # search according to p
        p *= game.get_valid_moves(board)
        p /= torch.sum(p)
        action = torch.argmax(p).numpy()

        # search according to v
        # action, best_v = 0, 1
        # for a, valid in enumerate(game.get_valid_moves(board)):
        #     if valid:
        #         t_board, t_player = game.get_next_state(board, player, a)
        #
        #         if player == 1:
        #             t_p, t_v = policy1(torch.tensor(game.get_player_pov(t_board, t_player), dtype=torch.float32))
        #         else:
        #             t_p, t_v = policy2(torch.tensor(game.get_player_pov(t_board, t_player), dtype=torch.float32))
        #
        #         if t_v < best_v:
        #             best_v = t_v
        #             action = a

        board, player = game.get_next_state(board, player, action)

        # render
        if render:
            game.render(board)

        # check game end
        done = game.get_game_ended(board, 1)

    return done
