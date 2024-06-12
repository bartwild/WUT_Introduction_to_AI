import mplcursors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import Counter


def animate_route(data):
    """
    Animates a route on a 2D plot.
    Args:
    - data: a tuple containing a list of (x, y) coordinates representing the route and a float representing the distance of the route.
    Returns:
    - None
    """
    route, dist = data
    print("plotting for dist:", dist)
    fig, ax1 = plt.subplots(figsize=(8, 8))

    # route plot
    x = [city[0] for city in route]
    y = [city[1] for city in route]
    x.append(route[0][0])
    y.append(route[0][1])
    line, = ax1.plot(x, y, marker='o', linestyle='-', color='b')

    def update(frame):
        if frame < len(route):
            x = [city[0] for city in route[:frame]]
            y = [city[1] for city in route[:frame]]
            x.append(route[frame][0])
            y.append(route[frame][1])
            line.set_data(x, y)
        else:
            x = [city[0] for city in route]
            y = [city[1] for city in route[:frame]]
            x.append(x[0])
            y.append(y[0])
            line.set_data(x, y)
    plt.xlabel("X")
    plt.ylabel("Y")
    _ = FuncAnimation(fig, update, frames=range(1, len(route) + 1), repeat=False)
    plt.show()


def plot_results(distances):
    """
    Plots the results of multiple runs of an algorithm.
    Parameters:
    - distances (list): A list of distances obtained from multiple runs of an algorithm.
    Returns:
    - None
    """
    counter = Counter(distances)
    distances, counts = zip(*sorted(counter.items()))

    _, ax = plt.subplots()
    bars = ax.bar(distances, counts)
    plt.xlabel("Długość trasy")
    plt.ylabel("Ilość wystąpień")
    plt.title("Wyniki wielokrotnego uruchomienia algorytmu")

    # add cursor
    cursor = mplcursors.cursor(bars)

    @cursor.connect("add")
    def on_add(sel):
        index = sel.target.index
        x = distances[index]
        y = counts[index]
        label = f"Długość: {x}, Ilość wystąpień: {y}"
        sel.annotation.set_text(label)

    plt.show()


def plot_generations(best_temp_generations):
    """
    Plots the best individuals from each generation.
    Args:
        best_temp_generations (dict): A dictionary containing the best individual from each generation.
    Returns:
        None
    """
    generations = len(best_temp_generations[0])
    runs = len(best_temp_generations)
    average_lengths = [sum(best_temp_generations[run][generation] for run in range(runs)) / runs for generation in range(generations)]
    plt.bar(range(generations), average_lengths)
    plt.xlabel("Generacja")
    plt.ylabel("Długość drogi")
    plt.title("Najlepszy osobnik w każdej generacji")
    plt.show()
