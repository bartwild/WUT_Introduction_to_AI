from visualization import compare_results_by_depth, visualize_tree_game
from game import Game, GameStatus
from utils import DEPTH, GAME_TYPE, INIT_POS_P1, INIT_POS_P2, DRAW_PARAMS, BOARD_SIZE
import random
import time


results = []
depth_list = []
# Simulate game
start = time.time()
"""for depth in range(1, 7):
    depth_list.append(depth)
    scores = [0, 0, 0]
    for i in range(100):
        init_pos_1 = random.randint(0, BOARD_SIZE-1), random.randint(0, BOARD_SIZE-1)
        init_pos_2 = random.randint(0, BOARD_SIZE-1), random.randint(0, BOARD_SIZE-1)
        while init_pos_1 == init_pos_2:
            init_pos_2 = random.randint(0, BOARD_SIZE-1), random.randint(0, BOARD_SIZE-1)
        game = Game(DRAW_PARAMS, init_pos_1, init_pos_2, depth, False, GAME_TYPE)
        result = game.simulate_game()
        if result == GameStatus.P1_WON:
            scores[0] += 1
        elif result == GameStatus.P2_WON:
            scores[1] += 1
        else:
            scores[2] += 1
    results.append(scores)
print(time.time() - start)
compare_results_by_depth(results, depth_list)
"""
#Simulate game by click
game = Game(DRAW_PARAMS, INIT_POS_P1, INIT_POS_P2, DEPTH, True,  GAME_TYPE)

game.run()

# visualize tree game
# game = Game(DRAW_PARAMS, INIT_POS_P1, INIT_POS_P2, DEPTH, False, GAME_TYPE)

# visualize_tree_game(game, f"minimax-{DEPTH}_depth.png")
