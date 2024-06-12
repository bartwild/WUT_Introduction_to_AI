from .board import Board
from .mini_max import mini_max
import random


def random_move(board):
    """
    Returns a random move for the current player on the given board.

    Args:
        board (Board): The current game board.

    Returns:
        tuple: A tuple representing the row and column of the randomly selected move.
    """
    if board.p1_turn:
        x, y = board.actual_pos_of_p1
    else:
        x, y = board.actual_pos_of_p2
    possible_moves = board.get_possible_moves(x, y)
    index = random.randint(0, len(possible_moves)-1)
    return (possible_moves[index].row, possible_moves[index].col)


def mini_max_move(board: Board, depth: int):
    """
    Returns the best move for the current player using the Mini-Max algorithm with alpha-beta pruning.

    Args:
        board (Board): The current game board.
        depth (int): The maximum depth to search in the game tree.

    Returns:
        Tuple[int, int]: The row and column of the best move.
    """
    moves = mini_max(board, depth)
    length = len(moves)
    if length < 2:
        return (moves[0][0].row, moves[0][0].col)
    else:
        index = random.randint(0, length-1)
        return (moves[index][0].row, moves[index][0].col)
