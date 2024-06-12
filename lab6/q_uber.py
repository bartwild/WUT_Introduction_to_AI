from random import randint, choice, uniform, seed
import numpy as np


def generate_lab(n, holes, start, end):
    """
    Generate a labyrinth with randomly placed holes.

    Args:
        n (int): The size of the labyrinth (n x n).
        holes (int): The number of holes to be placed in the labyrinth.

    Returns:
        list: A 2D list representing the labyrinth, where 0 represents a path and 1 represents a hole.
    """
    lab = [[0 for _ in range(n)] for _ in range(n)]
    made_holes = 0
    hole_positions = set()

    while made_holes < holes:
        x, y = randint(0, n-1), randint(0, n-1)

        if (x, y) in [start, end]:
            continue

        if (x, y) not in hole_positions:
            lab[x][y] = 1
            hole_positions.add((x, y))
            made_holes += 1
    return lab


class Node:
    """
    Represents a node in a grid.

    Attributes:
        x (int): The x-coordinate of the node.
        y (int): The y-coordinate of the node.
        lab (list): The grid representing the labyrinth.
        rows (int): The number of rows in the grid.
        kids (list): The available moves from the current node.
    """

    def __init__(self, x, y, lab):
        self.x = x
        self.y = y
        self.lab = lab
        self.rows = len(lab)
        self.kids = self.check_available_moves()

    def check_available_moves(self):
        """
        Checks and returns the available moves from the current node.

        Returns:
            list: A list of available moves as tuples (x, y).
        """
        moves = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < self.rows and 0 <= ny < self.rows and self.lab[nx][ny] == 0:
                moves.append((ny, nx))
        return moves


def dfs(lab):
    """
    Performs a depth-first search on a given labyrinth.

    Args:
        lab (list): The labyrinth represented as a 2D list.

    Returns:
        bool: True if there is a path from the starting point to the ending point in the labyrinth, False otherwise.
    """
    first = Node(0, 0, lab)
    last = Node(len(lab)-1, len(lab)-1, lab)
    possible_nodes = node_dfs([], lab, first)
    return any(node.x == last.x and node.y == last.y for node in possible_nodes)


def node_dfs(visited, lab, node):
    """
    Performs a depth-first search starting from the given node in a given labyrinth.

    Args:
        visited (list): A list of visited nodes.
        lab (list): The labyrinth represented as a 2D list.
        node (Node): The current node.

    Returns:
        list: A list of visited nodes.
    """
    if not any(node.x == n.x and node.y == n.y for n in visited):
        visited.append(node)
        for kid in node.check_available_moves():
            node_dfs(visited, lab, Node(kid[1], kid[0], lab))
    return visited


def save_lab_to_file(lab, file_name):
    """
    Save the lab to a file.

    Args:
        lab (list): The lab to be saved.
        file_name (str): The name of the file to save the lab to.
    """
    with open(file_name, 'w') as data:
        for row in lab:
            data.write(' '.join(map(str, row)) + '\n')


def read_lab_from_file(file_name):
    """
    Read a lab from a file and return it as a list of lists.

    Args:
        file_name (str): The name of the file to read.

    Returns:
        list: A list of lists representing the lab.

    """
    with open(file_name, 'r') as data:
        return [[int(num) for num in line.split()] for line in data]


class Musk_Taxi:
    """
    Simulation of an autonomous taxi in a labyrinth.

    Available moves:
    action 0 - left
    action 1 - up
    action 2 - right
    action 3 - down

    Rewards:
    driving outside the labyrinth: -10
    hitting a person: -10
    achieving goal: +20
    proper move: -1

    Attributes:
    - lab (list): The labyrinth represented as a 2D list.
    - rows (int): The number of rows in the labyrinth.
    - x (int): The current x-coordinate of the taxi.
    - y (int): The current y-coordinate of the taxi.
    - x_done (int): The x-coordinate of the target location.
    - y_done (int): The y-coordinate of the target location.
    - done (bool): Indicates if the taxi has reached the target location.
    - failed (bool): Indicates if the taxi has hit a person.

    Methods:
    - __init__(self, lab, target): Initializes the Musk_Taxi object.
    - make_action(self, action): Moves the taxi based on the given action and returns the corresponding reward.
    - drive(self, q_table, beta, gamma, epsilon): Drives the taxi using the Q-learning algorithm.
    - random_drive(self, random_table): Drives the taxi randomly.
    - no_of_states(self): Returns the total number of states in the labyrinth.
    - get_state(self): Returns the current state of the taxi.
    """

    def __init__(self, lab, target):
        self.rows = len(lab)
        self.lab = lab
        self.x = 0
        self.y = 0
        self.x_done = min(self.rows - 1, target[0])
        self.y_done = min(self.rows - 1, target[1])
        self.done = False
        self.failed = False

    def make_action(self, action):
        """
        Perform an action in the environment.

        Args:
            action (int): The action to be performed. Valid values are 0, 1, 2, or 3.

        Returns:
            int: The reward obtained after performing the action.
        """
        previous_position = self.x, self.y
        if action == 0:
            self.y -= 1
        elif action == 1:
            self.x -= 1
        elif action == 2:
            self.y += 1
        elif action == 3:
            self.x += 1
        if self.y == self.y_done and self.x == self.x_done:
            self.done = True
            return 20
        elif self.x == self.rows or self.y == self.rows or self.x < 0 or self.y < 0:
            self.x, self.y = previous_position
            return -10
        elif self.lab[self.x][self.y] == 1:
            self.done = True
            self.failed = True
            return -10
        else:
            return -1

    def drive(self, q_table, beta, gamma, epsilon):
        """
        Drives the agent using the Q-learning algorithm.

        Parameters:
        - q_table (numpy.ndarray): The Q-table used for action selection.
        - beta (float): The exploration rate.
        - gamma (float): The discount factor.
        - epsilon (float): The exploitation rate.

        Returns:
        - steps (list): The list of actions taken during the drive.
        - q_table (numpy.ndarray): The updated Q-table.
        """
        steps = []
        while not self.done:
            state = self.get_state()
            if uniform(0, 1) < epsilon:
                action = randint(0, 3)
            else:
                action = np.argmax(q_table[state])
            reward = self.make_action(action)
            steps.append(action)
            if reward == -10:
                if action % 2 == 0:
                    steps.append(action+1)
                else:
                    steps.append(action-1)
            next_state = self.get_state()
            q_table = update_q_table(q_table, state, action, beta, gamma, reward, next_state)
        return steps, q_table

    def random_drive(self, random_table):
        """
        Simulates a random drive by selecting actions randomly from a given random_table.

        Parameters:
        - random_table (dict): A dictionary containing the available actions for each state.

        Returns:
        - steps (list): A list of actions taken during the random drive.
        - random_table (dict): The updated random_table after removing actions that resulted in a reward of -10.
        """
        steps = []
        while not self.done:
            state = self.get_state()
            action = choice(random_table[state])
            reward = self.make_action(action)
            steps.append(action)
            if reward == -10:
                random_table[state].remove(action)
                if action % 2 == 0:
                    steps.append(action+1)
                else:
                    steps.append(action-1)
            if reward == 20:
                steps.append(4)
        return steps, random_table

    def num_of_states(self):
        """
        Returns the total number of states in the grid.

        Returns:
            int: The number of states in the grid.
        """
        return self.rows ** 2

    def get_state(self):
        """
        Returns the state of the object based on its current coordinates.

        The state is calculated by multiplying the x-coordinate with the number of rows
        and adding the y-coordinate.

        Returns:
            int: The state of the object.
        """
        state = self.x * self.rows + self.y
        return state


def generate_q_table(columns, rows):
    """
    Generate a Q-table with the specified number of columns and rows.

    Parameters:
    columns (int): The number of columns in the Q-table.
    rows (int): The number of rows in the Q-table.

    Returns:
    numpy.ndarray: The generated Q-table, initialized with zeros.
    """
    return np.zeros((columns, rows))


def generate_random_table(states):
    """
    Generates a random table with the specified number of states.

    Parameters:
    states (int): The number of states in the table.

    Returns:
    list: A random table with the specified number of states.
    """
    random_table = []
    for _ in range(states):
        random_table.append([0, 1, 2, 3])
    return random_table


def update_q_table(q_table, state, action, beta, gamma, reward, next_state):
    """
    Update the Q-table based on the given state, action, reward, and next state.

    Parameters:
    q_table (numpy.ndarray): The Q-table to be updated.
    state (int): The current state.
    action (int): The action taken in the current state.
    beta (float): The learning rate.
    gamma (float): The discount factor.
    reward (float): The reward received for taking the action in the current state.
    next_state (int): The next state after taking the action.

    Returns:
    numpy.ndarray: The updated Q-table.
    """
    next_state_max = np.max(q_table[next_state])
    q_table[state, action] += beta * (reward + gamma * next_state_max - q_table[state, action])
    return q_table


def main():
    epsilon = 0.01
    gamma = 0.9
    beta = 1.0

    # lab = generate_lab(8, 20, (0, 0), (7, 0))
    # while not dfs(lab):
    #     lab = generate_lab(8, 20, (0, 0), (7, 0))
    # save_lab_to_file(lab, 'test2.txt')

    lab = read_lab_from_file('test2.txt')

    steps = [0]
    random_table = generate_random_table(64)
    iteration_no = 0
    # interval_steps = []
    while steps[-1] != 4:
        taxi = Musk_Taxi(lab, (7, 0))
        steps = []
        steps, random_table = taxi.random_drive(random_table)
        iteration_no += 1
    iters_needed = iteration_no
    best_steps = steps
    for _ in range(1000):
        taxi = Musk_Taxi(lab, (7, 0))
        steps = []
        steps, random_table = taxi.random_drive(random_table)
        iteration_no += 1
        if len(steps) < len(best_steps) and taxi.failed is False:
            best_steps = steps
            iters_needed = iteration_no
    # print(interval_steps)
    print(best_steps[0:-1])
    print("Length of best found route: ", len(best_steps)-1)
    print("Iterations needed to find random route: ", iters_needed)

    steps = []
    q_table = generate_q_table(64, 4)
    iteration_no = 0
    iters_needed = 0
    seed(2137)
    for _ in range(100):
        taxi = Musk_Taxi(lab, (7, 0))
        steps, q_table = taxi.drive(q_table, beta, gamma, epsilon)
        iteration_no += 1
        if len(steps) < len(best_steps) and taxi.failed is False:
            best_steps = steps
            iters_needed = iteration_no
    print("Iterations needed to find q-uber route: ", iters_needed)
    print(best_steps)


if __name__ == "__main__":
    main()
