

import numpy as np


class TicTacToe:
    """
    Game Logic for TicTacToe

    Player always starts as 1

    Board is a numpy array
    """

    def __init__(self):
        pass

    @staticmethod
    def get_init_board():
        return np.zeros((3, 3))

    @staticmethod
    def get_state_shape():
        return (3, 3)

    @staticmethod
    def get_action_size():
        return 3 * 3

    @staticmethod
    def get_next_state(board, player, action):
        board = board.flatten()
        board[action] = player
        board = board.reshape(3, 3)

        return board, -player

    @staticmethod
    def get_valid_moves(board):
        return np.where(board.flatten() == 0, 1, 0)

    @staticmethod
    def get_player_pov(board, player):
        return player * board

    @staticmethod
    def get_rep(board):
        return board.tostring()

    @staticmethod
    def get_game_ended(board, player):
        """return 1 if player won, 0 if player drew, and -1 if player lost"""

        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] != 0:
                return player if board[i][0] == player else -player

            if board[0][i] == board[1][i] == board[2][i] != 0:
                return player if board[i][0] == player else -player

        if board[0][0] == board[1][1] == board[2][2] != 0:
            return player if board[i][0] == player else -player

        if board[0][2] == board[1][1] == board[2][0] != 0:
            return player if board[i][0] == player else -player

        # if board is full return draw
        if not np.any(board == 0):
            return 0

        # game didn't end
        return None

    @staticmethod
    def render(board):
        print("-------------")
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i][j] == 1:
                    symbol = "X"
                elif board[i][j] == -1:
                    symbol = "O"
                else:
                    symbol = "_"

                print(symbol, end=" ")
            print()


if __name__ == "__main__":
    game = TicTacToe()
    board = game.get_init_board()
    player = 1

    for i in range(7):
      game.render(board)
      board, player = game.get_next_state(board, player, i)

    game.render(board)

