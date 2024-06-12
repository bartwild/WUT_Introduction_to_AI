from .board import Board
from .utils import GameStatus, Players


def score_evaluation(board):
    """
    Evaluates the score of the current board state.

    Args:
        board (Board): The current board state.

    Returns:
        int: The score of the current board state.
    """
    result = board.get_result()
    if result == GameStatus.DRAW:
        return 0
    if result == GameStatus.P1_WON:
        return 100
    if result == GameStatus.P2_WON:
        return -100
    if result == GameStatus.UNFINISHED:
        x1, y1 = board.actual_pos_of_p1
        x2, y2 = board.actual_pos_of_p2
        possible_moves_1 = board.get_possible_moves(x1, y1)
        possible_moves_2 = board.get_possible_moves(x2, y2)
        return len(possible_moves_1) - len(possible_moves_2)


def mini_max(board, depth):
    """
    This function implements the Mini-Max algorithm for the Isolation game.
    It recursively evaluates all possible moves up to a certain depth and returns the best move(s) for the current player.

    Args:
        board (Board): The current state of the game board.
        depth (int): The depth of the search tree to evaluate.

    Returns:
        list: A list of tuples representing the best move(s) for the current player.
              Each tuple contains a Field object representing the move and an integer representing the score of the move.
    """
    if board.is_end() or depth == 0:
        score = score_evaluation(board)
        return [(None, score)]

    scores = []
    possible_moves = []
    if board.p1_turn:
        row, col = board.actual_pos_of_p1
        possible_moves = board.get_possible_moves(row, col)
    else:
        row, col = board.actual_pos_of_p2
        possible_moves = board.get_possible_moves(row, col)
    for _, field in enumerate(possible_moves):
        new_board = board.copy()
        new_board.move(field.row, field.col)
        new_scores_list = mini_max(new_board, depth-1)
        new_score = new_scores_list[0]
        scores.append((field, new_score[1]))
    if board.p1_turn:
        target_score = max(scores, key=lambda k: k[1])
    else:
        target_score = min(scores, key=lambda k: k[1])
    return [score for score in scores if target_score[1] == score[1]]


def mini_max_with_tree(board: Board, depth: int, tree={}):
    """
    This function implements the MiniMax algorithm with a tree structure to find the best move for a player in a game of Isolation.

    Args:
    - board (Board): the current state of the game board
    - depth (int): the maximum depth of the search tree
    - tree (dict): a dictionary representing the search tree

    Returns:
    - score_list (list): a list of tuples representing the possible moves and their corresponding scores
    """
    tree["depth"] = depth
    tree["is_end"] = False
    tree["children"] = []
    if board.is_end() or depth == 0:
        tree["is_end"] = True
        score = score_evaluation(board)
        tree["best_score"] = [(None, score)]
        return [(None, score)]

    scores = []
    possible_moves = []
    if board.p1_turn:
        row, col = board.actual_pos_of_p1
        possible_moves = board.get_possible_moves(row, col)
        tree["player_turn"] = Players.P1
    else:
        row, col = board.actual_pos_of_p2
        possible_moves = board.get_possible_moves(row, col)
        tree["player_turn"] = Players.P2
    for _, field in enumerate(possible_moves):
        new_board = board.copy()
        new_board.move(field.row, field.col)
        children_tree = {}
        children_tree['field'] = field
        tree['children'].append(children_tree)
        new_scores_list = mini_max_with_tree(new_board, depth-1, children_tree)
        new_score = new_scores_list[0]
        scores.append((field, new_score[1]))
    if board.p1_turn:
        target_score = max(scores, key=lambda k: k[1])
    else:
        target_score = min(scores, key=lambda k: k[1])
    score_list = [score for score in scores if target_score[1] == score[1]]
    tree['best_score'] = score_list
    return score_list
