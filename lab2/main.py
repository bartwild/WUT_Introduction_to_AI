import random
import math
import time
import multiprocessing
import statistics
from plotting_utils import plot_results, animate_route, plot_generations
from utils import generate_cities, POPULATION_SIZE, MUTATION_RATE, TOURNAMENT_SIZE, NUM_GENERATIONS


def fitness(individual):
    """
    Calculates the total distance traveled by an individual in a TSP problem.
    Args:
        individual (list): A list of tuples representing the cities in the order they are visited.
    Returns:
        float: The total distance traveled by the individual.
    """
    distance = 0
    for i in range(len(individual)):
        city1 = individual[i]
        city2 = individual[(i + 1) % len(individual)]  # Wrap around to the first city
        distance += math.sqrt((city2[0] - city1[0]) ** 2 + (city2[1] - city1[1]) ** 2)
    return distance


def mutate(individual):
    """
    Mutates an individual by swapping two randomly selected genes with a probability of MUTATION_RATE. It's to change but the most efficient way.
    Args:
        individual (list): The individual to mutate.
    Returns:
        list: The mutated individual.
    """
    if random.random() < MUTATION_RATE:
        # amount = random.randint(0, len(individual)-1)
        amount = 1
        for _ in range(amount):
            index1 = random.randint(0, len(individual) - 1)
            index2 = random.randint(0, len(individual) - 1)
            individual[index1], individual[index2] = individual[index2], individual[index1]
    return individual


def tournament_selection(population):
    """
    Selects an individual from the population using tournament selection.
    Args:
        population (list): A list of individuals to select from.
    Returns:
        An individual selected using tournament selection.
    """
    tournament = random.sample(population, TOURNAMENT_SIZE)
    return min(tournament, key=fitness)


def compute(cities) -> tuple[list[int], float]:
    """
    Runs a genetic algorithm to find the shortest path through a set of cities.
    Returns:
        Tuple[List[int], float]: A tuple containing the best individual (a list of city IDs) and its fitness score.
    """
    # Define the initial population
    population = []
    for i in range(POPULATION_SIZE):
        individual = cities.copy()
        random.shuffle(individual)
        population.append(individual)

    best_individual = min(population, key=fitness)
    best_temp_runs = []
    # Run the genetic algorithm
    for i in range(NUM_GENERATIONS):
        # Select the parents
        parents = [tournament_selection(population) for _ in range(POPULATION_SIZE)]
        # Create the offspring
        offspring = []
        for i in range(0, POPULATION_SIZE):
            parent = parents[i]
            child = mutate(parent.copy())
            offspring.append(child)
        # Replace the population with the offspring
        population = offspring
        best_temp = min(population, key=fitness)
        best_temp_runs.append(fitness(best_temp))
        if fitness(best_individual) > fitness(best_temp):
            best_individual = best_temp
    print('Fitness:', fitness(best_individual))
    return best_individual, fitness(best_individual), best_temp_runs


def compute_once(args):
    """
    Computes the best and distance values once. Used for multithreading
    Args:
    - args: arguments to be passed to the compute function. Unsused currently.
    Returns:
    - A tuple containing the best and distance values.
    """
    cities = generate_cities("random")
    best, dist, gens = compute(cities)
    return best, dist, gens


if __name__ == '__main__':
    distances = []
    best_temp_generations = []
    start_time = time.time()
    with multiprocessing.Pool() as pool:
        results = pool.map(compute_once, range(12))
    end_time = time.time()
    execution_time = end_time - start_time
    print("Time of parallel execution:", execution_time)
    for result in results:
        distances.append(result[1])
        best_temp_generations.append(result[2])
    avg_dist = sum(distances) / len(distances)
    std_dev = statistics.stdev(distances)
    print("Avg dist = " + str(avg_dist))
    print("Standard deviation = " + str(std_dev))
    plot_results(distances)
    animate_route(sorted(results, key=lambda x: (x[1]))[0][:2])
    plot_generations(best_temp_generations)
