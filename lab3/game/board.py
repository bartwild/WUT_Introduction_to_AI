from .fields import AvailableField, P1Field, P2Field, BlockedField
from .utils import GameStatus
import numpy as np


class Board:
    """
    Represents the game board for the Isolation game.

    Attributes:
    - window: the Pygame window object
    - draw_params: the DrawParams object containing the parameters for drawing the board
    - actual_pos_of_p1: a list containing the current position of player 1 on the board
    - actual_pos_of_p2: a list containing the current position of player 2 on the board
    - p1_turn: a boolean indicating whether it is player 1's turn
    - p2_turn: a boolean indicating whether it is player 2's turn
    - board: a numpy array representing the game board
    """
    def __init__(self, window, draw_params, init_p1_pos, init_p2_pos):
        """
        Initializes the Board object.

        Parameters:
        - window: the Pygame window object
        - draw_params: the DrawParams object containing the parameters for drawing the board
        - init_p1_pos: a list containing the initial position of player 1 on the board
        - init_p2_pos: a list containing the initial position of player 2 on the board
        """
        self.board = np.full((draw_params.num_of_fields, draw_params.num_of_fields), None)
        self.draw_params = draw_params
        self.window = window
        self.actual_pos_of_p1 = init_p1_pos
        self.actual_pos_of_p2 = init_p2_pos
        self.p1_turn = True
        self.p2_turn = False
        self.__set_board(init_p1_pos, init_p2_pos)

    def __set_board(self, init_p1_pos, init_p2_pos):
        """
        Sets up the game board with available fields and player positions.

        Args:
            init_p1_pos (tuple): Initial position of player 1 as a tuple of (row, col).
            init_p2_pos (tuple): Initial position of player 2 as a tuple of (row, col).
        """
        for row in range(self.draw_params.num_of_fields):
            for col in range(self.draw_params.num_of_fields):
                self.board[row][col] = AvailableField(self.window, row, col, self.draw_params)

        row_p1, col_p1 = init_p1_pos
        row_p2, col_p2 = init_p2_pos
        self.board[row_p1][col_p1] = P1Field(self.window, row_p1, col_p1, self.draw_params)
        self.board[row_p2][col_p2] = P2Field(self.window, row_p2, col_p2, self.draw_params)

    def copy(self):
        new_board = Board(self.window, self.draw_params, self.actual_pos_of_p1, self.actual_pos_of_p2)
        new_board.board = np.copy(self.board)
        new_board.p1_turn, new_board.p2_turn = self.p1_turn, self.p2_turn
        return new_board

    def draw(self):
        for row in range(self.draw_params.num_of_fields):
            for col in range(self.draw_params.num_of_fields):
                if not self.board[row][col].is_drawed:
                    self.board[row][col].draw()

    def get_neighbors_fields(self, rowNumber, colNumber):
        """
        Returns a list of all neighboring fields of the given field.

        Args:
            rowNumber (int): The row number of the field.
            colNumber (int): The column number of the field.

        Returns:
            list: A list of all neighboring fields of the given field.
        """
        result = []
        for rowAdd in range(-1, 2):
            newRow = rowNumber + rowAdd
            if 0 <= newRow < self.draw_params.num_of_fields:
                for colAdd in range(-1, 2):
                    newCol = colNumber + colAdd
                    if 0 <= newCol < self.draw_params.num_of_fields and not (newCol == colNumber and newRow == rowNumber):
                        result.append(self.board[newRow][newCol])
        return result

    def get_possible_moves(self, rowNumber, colNumber):
        """
        Returns a list of possible moves for a given position on the board.

        Args:
        - rowNumber (int): the row number of the position
        - colNumber (int): the column number of the position

        Returns:
        - possible_moves (list): a list of AvailableField objects representing the possible moves
        """
        moves = self.get_neighbors_fields(rowNumber, colNumber)
        possible_moves = [field for field in moves if isinstance(field, AvailableField)]
        return possible_moves

    def move(self, rowNumber, colNumber):
        """
        Moves the player to the specified row and column number on the board.

        Args:
            rowNumber (int): The row number to move to.
            colNumber (int): The column number to move to.
        """
        current_pos = self.actual_pos_of_p1 if self.p1_turn else self.actual_pos_of_p2
        new_field = P1Field(self.window, rowNumber, colNumber, self.draw_params) if self.p1_turn else P2Field(self.window, rowNumber, colNumber, self.draw_params)
        self.board[current_pos[0]][current_pos[1]] = BlockedField(self.window, current_pos[0], current_pos[1], self.draw_params)
        self.board[rowNumber][colNumber] = new_field
        if self.p1_turn:
            self.actual_pos_of_p1 = [rowNumber, colNumber]
        else:
            self.actual_pos_of_p2 = [rowNumber, colNumber]
        self.p1_turn, self.p2_turn = self.p2_turn, self.p1_turn

    def is_end(self):
        """
        Check if the game has ended.

        Returns:
            bool: True if the game has ended, False otherwise.
        """
        if self.p1_turn:
            possible_moves = self.get_possible_moves(*self.actual_pos_of_p1)
        else:
            possible_moves = self.get_possible_moves(*self.actual_pos_of_p2)
        return len(possible_moves) == 0

    def get_result(self):
        """
        Determines the result of the game based on the current positions of the players and their possible moves.

        Returns:
            GameStatus: The status of the game (P1_WON, P2_WON, DRAW, or UNFINISHED).
        """
        x1, y1 = self.actual_pos_of_p1
        x2, y2 = self.actual_pos_of_p2
        possible_moves_of_p1 = self.get_possible_moves(x1, y1)
        possible_moves_of_p2 = self.get_possible_moves(x2, y2)
        num_of_possible_moves_fp = len(possible_moves_of_p1)
        num_of_possible_moves_sp = len(possible_moves_of_p2)
        if num_of_possible_moves_fp == 0 and num_of_possible_moves_sp == 0:
            return GameStatus.DRAW
        elif num_of_possible_moves_fp > 0 and num_of_possible_moves_sp > 0:
            return GameStatus.UNFINISHED
        elif num_of_possible_moves_fp == 0:
            return GameStatus.P2_WON
        else:
            return GameStatus.P1_WON
