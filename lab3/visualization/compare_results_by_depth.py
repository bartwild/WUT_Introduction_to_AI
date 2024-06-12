import matplotlib.pyplot as plt
import numpy as np


def compare_results_by_depth(results: list[int, int, int], depth_list: list[int]):
    """
    Plots a bar chart comparing the results of two players across different depths.

    Args:
    results (list): A list of tuples containing the number of wins for player 1, player 2, and draws at each depth.
    depth_list (list): A list of integers representing the different depths.

    Returns:
    None
    """
    labels = depth_list
    p1_win = []
    p2_win = []
    draws = []
    for result in results:
        p1_win.append(result[0])
        p2_win.append(result[1])
        draws.append(result[2])
    x = np.arange(len(labels))
    w = 0.20
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - w, p1_win, w, label='First player')
    bar2 = ax.bar(x, p2_win, w, label='Second player')
    bar3 = ax.bar(x + w, draws, w, label="Draws")
    ax.set_ylabel('Number of occurrences')
    ax.set_xticks(x, labels)
    ax.legend()
    ax.bar_label(bar1, padding=3)
    ax.bar_label(bar2, padding=3)
    ax.bar_label(bar3, padding=3)
    fig.tight_layout()
    plt.show()
