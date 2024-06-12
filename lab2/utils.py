import random
import numpy as np

myrandom = random.Random(42)
N = 20  # Number of cities
POPULATION_SIZE = 10*N
MUTATION_RATE = 0.4
TOURNAMENT_SIZE = 2
NUM_GENERATIONS = 5000


def generate_cities(type):
    """
    Generate a list of cities based on the given type.
    Args:
        type (str): The type of cities to generate. Can be "random", "checkboard", or "gauss".
    Returns:
        list: A list of tuples representing the coordinates of the generated cities.
    """
    if type == "random":
        return [(myrandom.uniform(0, 100), myrandom.uniform(0, 100)) for _ in range(N)]
    elif type == "checkboard":
        return [(x, y) for x in range(0, int(N / 5) + 1) for y in range(0, int(N / 10) + 1)]
    elif type == "gauss":
        num_clusters = 5
        cluster_radius = 10
        cities = []
        cities_generated = 0
        for _ in range(num_clusters):
            cluster_center = (random.uniform(0, 100), random.uniform(0, 100))
            remaining_cities = N - cities_generated
            cluster_size = min(remaining_cities, random.randint(1, 50))
            for _ in range(cluster_size):
                x = np.random.normal(cluster_center[0], cluster_radius)
                y = np.random.normal(cluster_center[1], cluster_radius)
                x = max(0, min(x, 100))
                y = max(0, min(y, 100))
                cities.append((x, y))
                cities_generated += 1
                if cities_generated >= N:
                    break
        return cities
