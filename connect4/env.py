"""
env.py — The Connect Four environment.

Same role as tictactoe/env.py — the rulebook. The agent never sees this
code, it only receives states, rewards, and done signals.

Board layout: 6 rows × 7 columns.
Pieces fall due to gravity — you choose a column, the piece lands in the
lowest empty row of that column.

Cell values: 0 = empty, 1 = player 1, 2 = player 2

The state passed to the neural network is the board flattened to 42 values:
    row 0 (top), row 1, ... row 5 (bottom) — left to right within each row.
"""

ROWS = 6
COLS = 7
WIN_LENGTH = 4


class ConnectFourEnv:

    def __init__(self):
        self.board = None
        self.current_player = None
        self.reset()

    def reset(self):
        """Start a fresh game. Returns the initial state."""
        self.board = [[0] * COLS for _ in range(ROWS)]
        self.current_player = 1
        return self._state()

    def step(self, col):
        """
        Drop the current player's piece into column `col` (0–6).
        The piece falls to the lowest empty row in that column.

        Returns:
          new_state  : flattened board (tuple of 42 ints)
          reward     : +1 if the player who just moved won, 0 otherwise
          done       : True if game is over (win or draw)
          winner     : 1, 2, or 0
        """
        if col not in self.legal_actions():
            raise ValueError(f"Column {col} is full or invalid.")

        # Find the lowest empty row in this column (gravity)
        row = self._landing_row(col)
        self.board[row][col] = self.current_player

        winner = self._check_winner(row, col)
        board_full = not any(self.board[0][c] == 0 for c in range(COLS))
        done = winner != 0 or board_full

        reward = 1 if winner == self.current_player else 0

        if not done:
            self.current_player = 3 - self.current_player

        return self._state(), reward, done, winner

    def legal_actions(self):
        """Return columns that still have space (top row is empty)."""
        return [c for c in range(COLS) if self.board[0][c] == 0]

    def render(self):
        """Print the board to the console, top row first."""
        symbols = {0: '.', 1: 'X', 2: 'O'}
        for row in self.board:
            print(' '.join(symbols[cell] for cell in row))
        print(' '.join(str(c) for c in range(COLS)))
        print()

    def _landing_row(self, col):
        """Find the lowest empty row in a column (where the piece lands)."""
        for row in range(ROWS - 1, -1, -1):
            if self.board[row][col] == 0:
                return row
        raise ValueError(f"Column {col} is full.")

    def _state(self):
        """
        Flatten the 2D board into a tuple of 42 values.
        Tuples are immutable — safe to use as dict keys and tensor inputs.
        """
        return tuple(cell for row in self.board for cell in row)

    def _check_winner(self, last_row, last_col):
        """
        Check whether the last move created a win.
        Only checks lines passing through (last_row, last_col) — efficient
        because a win can only be created by the piece just placed.

        Returns the winning player (1 or 2), or 0 if no winner.
        """
        player = self.board[last_row][last_col]

        # All four directions to check: horizontal, vertical, two diagonals
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1  # count the piece just placed

            # Look in the positive direction
            r, c = last_row + dr, last_col + dc
            while 0 <= r < ROWS and 0 <= c < COLS and self.board[r][c] == player:
                count += 1
                r += dr
                c += dc

            # Look in the negative direction
            r, c = last_row - dr, last_col - dc
            while 0 <= r < ROWS and 0 <= c < COLS and self.board[r][c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= WIN_LENGTH:
                return player

        return 0
