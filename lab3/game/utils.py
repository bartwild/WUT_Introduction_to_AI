from enum import Enum


Color = tuple[int, int, int]


class DrawParameters:
    """
    A class representing the parameters used for drawing the game board.

    Attributes:
    - window_size (int): The size of the game window in pixels.
    - num_of_fields (int): The num of fields on the game board.
    - block_field_color (tuple): The RGB color of blocked fields.
    - available_field_color (tuple): The RGB color of available fields.
    - p1_color (tuple): The RGB color of player 1's pieces.
    - p2_color (tuple): The RGB color of player 2's pieces.
    """
    def __init__(self, window_size, num_of_fields, block_field_color, available_field_color, p1_color, p2_color):
        self.window_h = self.window_w = window_size
        self.num_of_fields = num_of_fields
        self.field_size = window_size // num_of_fields
        self.player_size = self.field_size / 2 - 8
        self.padding = self.field_size // 8
        self.block_field_color = block_field_color
        self.available_field_color = available_field_color
        self.p1_color = p1_color
        self.p2_color = p2_color


class Players(Enum):
    P1 = 1
    P2 = 2


class GameStatus(Enum):
    P1_WON = 1
    P2_WON = 2
    DRAW = 3
    UNFINISHED = 4


class GameType(Enum):
    RANDOM_AND_RANDOM = 1
    RANDOM_AND_MINIMAX = 2
    MINIMAX_AND_MINIMAX = 3
