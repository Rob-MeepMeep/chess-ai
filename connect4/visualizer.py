"""
visualizer.py — pygame Connect Four board display for HAL-3000 demo games.

Opens a window and draws the board state after each move.
Used only during periodic demo games, not during the main training loop.

Board layout:
    Blue grid, yellow pieces for HAL-1, red pieces for HAL-2.
    Classic Connect Four colours.
"""

import time
import pygame

ROWS = 6
COLS = 7

CELL   = 90    # pixels per board cell
RADIUS = 35    # piece radius in pixels
HEADER = 70    # height of the info strip above the board

WIDTH  = COLS * CELL           # 630px
HEIGHT = ROWS * CELL + HEADER  # 610px

# Colours
C_BG    = (15,  15,  15)   # window background
C_BOARD = (30,  90, 200)   # board blue
C_EMPTY = (20,  20,  40)   # empty slot
C_P1    = (240, 200,  30)  # HAL-1: yellow
C_P2    = (220,  50,  50)  # HAL-2: red
C_TEXT  = (240, 240, 240)  # white-ish text


class Visualizer:

    def __init__(self, title="HAL-3000 — Connect Four"):
        pygame.init()
        self.screen   = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(title)
        self.font_lg  = pygame.font.SysFont("Arial", 28, bold=True)
        self.font_sm  = pygame.font.SysFont("Arial", 18)
        self.clock    = pygame.time.Clock()

    def draw(self, state, episode=None, label=None, delay=0.5):
        """
        Draw the board and pause so the move is visible.

        state:   42-tuple from the env (0=empty, 1=HAL-1, 2=HAL-2)
        episode: shown top-left if provided
        label:   status text shown in the header centre
        delay:   seconds to hold the frame before returning
        """
        self._handle_events()
        self.screen.fill(C_BG)
        self._draw_header(episode, label)
        self._draw_board(state)
        pygame.display.flip()
        time.sleep(delay)

    def show_result(self, winner, delay=2.5, screenshot_path=None):
        """Overlay the result on the current board and hold.

        If screenshot_path is given, saves a PNG of this frame before closing.
        """
        self._handle_events()

        if winner == 1:
            text, colour = "HAL-1 wins!", C_P1
        elif winner == 2:
            text, colour = "HAL-2 wins!", C_P2
        else:
            text, colour = "Draw", C_TEXT

        surf = self.font_lg.render(text, True, colour)
        rect = surf.get_rect(center=(WIDTH // 2, HEIGHT // 2))

        # Dark backing rectangle so text is readable over the board
        bg = pygame.Surface((rect.width + 40, rect.height + 20), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 190))
        self.screen.blit(bg, (rect.x - 20, rect.y - 10))
        self.screen.blit(surf, rect)
        pygame.display.flip()

        if screenshot_path:
            pygame.image.save(self.screen, screenshot_path)

        time.sleep(delay)

    def close(self):
        pygame.quit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_header(self, episode, label):
        # Episode counter — top left
        if episode is not None:
            ep = self.font_sm.render(f"Episode {episode:,}", True, C_TEXT)
            self.screen.blit(ep, (12, HEADER // 2 - ep.get_height() // 2))

        # Status label — centre
        if label:
            lbl = self.font_sm.render(label, True, C_TEXT)
            self.screen.blit(lbl, lbl.get_rect(center=(WIDTH // 2, HEADER // 2)))

        # Colour key — top right (coloured text, no extra circles needed)
        k2 = self.font_sm.render("HAL-2", True, C_P2)
        k1 = self.font_sm.render("HAL-1", True, C_P1)
        self.screen.blit(k2, (WIDTH - k2.get_width() - 10, HEADER // 2 - k2.get_height() // 2))
        self.screen.blit(k1, (WIDTH - k2.get_width() - k1.get_width() - 22, HEADER // 2 - k1.get_height() // 2))

    def _draw_board(self, state):
        # Blue board background
        pygame.draw.rect(self.screen, C_BOARD, (0, HEADER, WIDTH, ROWS * CELL))

        piece_colours = {0: C_EMPTY, 1: C_P1, 2: C_P2}

        for idx, value in enumerate(state):
            row = idx // COLS
            col = idx  % COLS
            cx  = col * CELL + CELL // 2
            cy  = HEADER + row * CELL + CELL // 2
            pygame.draw.circle(self.screen, piece_colours[value], (cx, cy), RADIUS)

    def _handle_events(self):
        """Process OS events so the window stays responsive."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit
