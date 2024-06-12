from .board import Board
from .moves import random_move, mini_max_move
from .utils import GameType
import pygame


class Game:
    """
    Represents a game of Isolation.

    Parameters:
    draw_params (DrawParameters): The params for drawing the game board.
    init_p1_pos (tuple): The init pos of player 1.
    init_p2_pos (tuple): The init pos of player 2.
    depth (int): The depth of the minimax algorithm.
    is_displaying (bool): Whether or not to display the game window.
    gametype (GameType): The type of game to play.

    Attributes:
    window (pygame.Surface): The game window.
    is_running (bool): Whether or not the game is running.
    board (Board): The game board.
    depth (int): The depth of the minimax algorithm.
    gametype (GameType): The type of game to play.
    """
    def __init__(self, draw_params, init_p1_pos, init_p2_pos, depth, is_displaying=True, gametype=GameType.RANDOM_AND_RANDOM):
        self.window, self.depth, self.is_running, self.gametype = None, depth, True, gametype
        if is_displaying:
            self.window = pygame.display.set_mode((draw_params.window_w, draw_params.window_h))
        self.board = Board(self.window, draw_params, init_p1_pos, init_p2_pos)

    def update(self):
        self.board.draw()
        pygame.display.update()

    def run(self):
        """
        Runs the game loop until the game is over or the user quits.

        If the game is over, prints the result of the game.
        """
        while self.is_running:
            if self.board.is_end():
                self.is_running = False
                print(self.board.get_result())
                break
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.board.p1_turn:
                        row, col = self.move_depent_on_game_type(self.gametype, 1)
                    else:
                        row, col = self.move_depent_on_game_type(self.gametype, 2)
                    self.board.move(row, col)
            self.update()
        pygame.quit()

    def simulate_game(self):
        """
        Simulates a game of isolation until it ends and returns the result.

        Returns:
            The result of the game (either "player1" or "player2").
        """
        result = None
        while self.is_running:
            if self.board.is_end():
                self.is_running = False
                result = self.board.get_result()
                break
            if self.board.p1_turn:
                row, col = self.move_depent_on_game_type(self.gametype, 1)
            else:
                row, col = self.move_depent_on_game_type(self.gametype, 2)
            self.board.move(row, col)
        return result

    def move_depent_on_game_type(self, gametype, player):
        """
        Returns a move based on the game type and player.

        Args:
            gametype (GameType): The type of game being played.
            player (int): The player making the move.

        Returns:
            tuple: The move to be made.
        """
        if gametype == GameType.RANDOM_AND_RANDOM or (player == 1 and gametype == GameType.RANDOM_AND_MINIMAX):
            return random_move(self.board)
        elif (gametype == GameType.RANDOM_AND_MINIMAX and player == 2) or gametype == GameType.MINIMAX_AND_MINIMAX:
            return mini_max_move(self.board, self.depth)
