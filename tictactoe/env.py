"""
env.py — The Tic-Tac-Toe environment.

This file contains the rules of the game. The agent doesn't know any of this —
it only ever sees states, takes actions, and receives rewards. The environment
is the referee; the agent is the player.

Board layout (cells 0–8):
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8

Cell values: 0 = empty, 1 = X, 2 = O
"""


class TicTacToeEnv:

    # All 8 ways to win: 3 rows, 3 columns, 2 diagonals
    WINS = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
        (0, 4, 8), (2, 4, 6),              # diagonals
    ]

    def __init__(self):
        self.board = None
        self.current_player = None
        self.reset()

    def reset(self):
        """Start a fresh game. Returns the initial state."""
        self.board = [0] * 9
        self.current_player = 1  # X always goes first
        return self._state()

    def step(self, action):
        """
        Place the current player's piece at cell `action` (0–8).

        Returns a tuple of four things:
          - new_state  : the board after the move (tuple of 9 ints)
          - reward     : +1 if the player who just moved won, 0 otherwise
          - done       : True if the game is over (win or draw)
          - winner     : 1 (X), 2 (O), or 0 (draw / still playing)
        """
        if self.board[action] != 0:
            raise ValueError(f"Cell {action} is already occupied.")

        self.board[action] = self.current_player

        winner = self._check_winner()
        board_full = all(cell != 0 for cell in self.board)
        done = winner != 0 or board_full

        reward = 1 if winner == self.current_player else 0

        # Switch turns — the trick: 1→2 and 2→1 both equal 3 - current
        if not done:
            self.current_player = 3 - self.current_player

        return self._state(), reward, done, winner

    def legal_actions(self):
        """Return the indices of all empty cells."""
        return [i for i, cell in enumerate(self.board) if cell == 0]

    def render(self):
        """Print a human-readable board to the console."""
        symbols = {0: '.', 1: 'X', 2: 'O'}
        for row in range(3):
            print(' | '.join(symbols[self.board[row * 3 + col]] for col in range(3)))
            if row < 2:
                print('---------')
        print()

    def _state(self):
        """
        Return the board as a tuple.
        Tuples are immutable, so Python can use them as dictionary keys —
        which is exactly what the Q-table in agent.py needs.
        """
        return tuple(self.board)

    def _check_winner(self):
        """Return 1 if X won, 2 if O won, 0 if nobody has won yet."""
        for a, b, c in self.WINS:
            if self.board[a] == self.board[b] == self.board[c] != 0:
                return self.board[a]
        return 0
