import matplotlib.pyplot as plt
from itertools import product
from q_uber import Musk_Taxi, generate_q_table, read_lab_from_file
import random


def test_parameters(lab, epsilons, betas, gammas, num_trials):
    """
    Test different combinations of parameters for a given lab environment.

    Args:
        lab (Lab): The lab environment to test.
        epsilons (list): List of epsilon values to test.
        betas (list): List of beta values to test.
        gammas (list): List of gamma values to test.
        num_trials (int): Number of trials to run for each parameter combination.

    Returns:
        dict: A dictionary containing the average iterations and success rate for each parameter combination.
    """
    results = {}

    for epsilon, beta, gamma in product(epsilons, betas, gammas):
        total_iterations = 0
        founds = 0
        random.seed(2137)
        for _ in range(num_trials):
            taxi = Musk_Taxi(lab, (7, 7))
            q_table = generate_q_table(taxi.num_of_states(), 4)
            iterations_needed = 0
            iterations_no = 0
            best_steps = [0] * 15
            for _ in range(1000):
                taxi = Musk_Taxi(lab, (7, 7))
                steps, q_table = taxi.drive(q_table, beta, gamma, epsilon)
                iterations_no += 1
                if len(steps) < len(best_steps) and taxi.failed is False:
                    best_steps = steps
                    iterations_needed = iterations_no
                    total_iterations += iterations_needed
                    founds += 1
                    break

        average_iterations = total_iterations / num_trials
        results[(epsilon, beta, gamma)] = (average_iterations, founds/num_trials)

    return results


# Define parameter ranges
alphas = [0.01]
betas = [1.0]
gammas = [0.0+0.01*i for i in range(100)]
num_trials = 10


lab = read_lab_from_file('test.txt')

# Run the tests
results = test_parameters(lab, alphas, betas, gammas, num_trials)

# Plotting the results
# This will generate multiple plots, one for each combination of alpha and gamma, showing the performance across different beta values.
for alpha, beta in product(alphas, betas):
    gamma_values = []
    iterations_needed = []
    for gamma in gammas:
        gamma_values.append(gamma)
        iterations_needed.append(results[(alpha, beta, gamma)][0])

    plt.figure()
    plt.plot(gamma_values, iterations_needed, marker='o')
    plt.title(f'Performance for eps={alpha}, beta={beta}')
    plt.xlabel('Gamma')
    plt.ylabel('Average Iterations Needed')
    plt.grid(True)
plt.show()

for alpha, beta in product(alphas, betas):
    gamma_values = []
    iterations_needed = []
    for gamma in gammas:
        gamma_values.append(gamma)
        iterations_needed.append(results[(alpha, beta, gamma)][1])

    plt.figure()
    plt.plot(gamma_values, iterations_needed, marker='o')
    plt.title(f'Performance for eps={alpha}, beta={beta}')
    plt.xlabel('Gamma')
    plt.ylabel('Found percentage')
    plt.grid(True)
plt.show()
