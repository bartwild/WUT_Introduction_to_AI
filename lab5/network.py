
import random
import numpy as np
import matplotlib.pyplot as plt


class Network:
    """
    A class representing a neural network.

    Attributes:
    - num_layers (int): The number of layers in the network.
    - sizes (list): A list of integers representing the number of neurons in each layer.
    - weights (list): A list of weight matrices for each layer.
    - biases (list): A list of bias vectors for each layer.

    Methods:
    - __init__(self, sizes: list): Initializes the network with the given sizes.
    - weight_init(self): Initializes the weights and biases of the network.
    - feedforward(self, input): Performs a feedforward pass through the network and returns the output.
    - stochastic_gradient_descent(self, training_data, epochs, batch_size, learn_rate, lambda_, test_data):
        Performs stochastic gradient descent to train the network.
    - update_network(self, mini_batch, learn_rate, lambda_, n): Updates the network's weights and biases using
        backpropagation and gradient descent.
    - backpropagation(self, image, digit): Performs backpropagation to compute the gradients of the weights and biases.
    - accuracy(self, data, convert=False): Computes the accuracy of the network on the given data.

    """

    def __init__(self, sizes: list):
        """
        Initializes the network with the given sizes.

        Args:
        - sizes (list): A list of integers representing the number of neurons in each layer.

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = None
        self.biases = None
        self.weight_init()

    def weight_init(self):
        """
        Initializes the weights and biases of the network.

        """
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]

    def feedforward(self, input, fun='sigmoid'):
        """
        Performs a feedforward pass through the network and returns the output.

        Args:
        - input: The input to the network.

        Returns:
        - The output of the network.

        """
        for bias, weight in zip(self.biases, self.weights):
            input = activation_fun(np.dot(weight, input)+bias, fun)
        return input

    def stochastic_gradient_descent(self, training_data, epochs, batch_size, learn_rate, lambda_, test_data, show_confusion_matrix=False, fun="sigmoid"):
        """
        Performs stochastic gradient descent to train the network.

        Args:
        - training_data: The training data.
        - epochs (int): The number of epochs to train for.
        - batch_size (int): The size of each mini-batch.
        - learn_rate (float): The learning rate.
        - lambda_ (float): The regularization parameter.
        - test_data: The test data.

        Returns:
        - The accuracy of the network on the test data.

        """
        training_data_len = len(training_data)

        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[j:j+batch_size] for j in range(0, training_data_len, batch_size)]
            for mini_batch in mini_batches:
                self.update_network(mini_batch, learn_rate, lambda_, len(training_data), fun)

            accuracy = self.accuracy(test_data, fun=fun)
            class_accuracy = self.class_accuracy(test_data, fun=fun)
            print(f'Epoch {i+1}. Accuracy: {accuracy}%, Class accuracy: {class_accuracy}')
            if i == epochs-1:
                if show_confusion_matrix:
                    self.class_confusion_matrix(test_data, fun=fun)
                return accuracy

    def update_network(self, mini_batch, learn_rate, lambda_, n, fun="sigmoid"):
        """
        Updates the network's weights and biases using backpropagation and gradient descent.

        Args:
        - mini_batch: The mini-batch of training data.
        - learn_rate (float): The learning rate.
        - lambda_ (float): The regularization parameter.
        - n (int): The total number of training examples.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for image, digit in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(image, digit, fun)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-learn_rate*(lambda_/n))*w-(learn_rate/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learn_rate/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backpropagation(self, image, digit, fun="sigmoid"):
        """
        Performs backpropagation to compute the gradients of the weights and biases.

        Args:
        - image: The input image.
        - digit: The target digit.

        Returns:
        - A tuple (nabla_b, nabla_w) containing the gradients of the biases and weights, respectively.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = image
        activations = [image]
        zs = []
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activation)+bias
            zs.append(z)
            activation = activation_fun(z, fun)
            activations.append(activation)
        delta = (activations[-1] - digit) * activation_prime(zs[-1], fun)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = activation_prime(z, fun)
            delta = np.dot(self.weights[1-layer].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-1-layer].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False, fun="sigmoid"):
        """
        Computes the accuracy of the network on the given data.

        Args:
        - data: The data to compute the accuracy on.
        - convert (bool): Whether to convert the network's output to one-hot encoding.

        Returns:
        - The accuracy of the network on the data.
        """
        if convert:
            results = [(np.argmax(self.feedforward(x, fun)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x, fun)), y) for (x, y) in data]
        result_accuracy = round(sum(int(x == y) for (x, y) in results)/len(data)*100, 2)
        return result_accuracy

    def class_accuracy(self, data, convert=False, fun="sigmoid"):
        """
        Computes the accuracy of the network on the given data for every class.

        Args:
        - data: The data to compute the accuracy on.
        - convert (bool): Whether to convert the network's output to one-hot encoding.

        Returns:
        - The accuracy of the network on the data per class.
        """
        if convert:
            results = [(np.argmax(self.feedforward(x, fun)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x, fun)), y) for (x, y) in data]
        class_accuracy = {}
        for (predicted_class, actual_class) in results:
            if actual_class not in class_accuracy:
                class_accuracy[actual_class] = {"correct": 0, "total": 0}
            class_accuracy[actual_class]["total"] += 1
            if predicted_class == actual_class:
                class_accuracy[actual_class]["correct"] += 1
        class_accuracy_percentage = {}
        for y, accuracy in class_accuracy.items():
            class_accuracy_percentage[y] = round(accuracy["correct"] / accuracy["total"] * 100, 2)

        return class_accuracy_percentage

    def class_confusion_matrix(self, data, convert=False, fun="sigmoid"):
        """
        Computes the confusion matrix.

        Args:
        - data: The data to compute the accuracy on.
        - convert (bool): Whether to convert the network's output to one-hot encoding.

        Returns:
        - Returns the confusion matrix.

        """
        if convert:
            results = [(np.argmax(self.feedforward(x, fun)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x, fun)), y) for (x, y) in data]
        confusion_matrix = {i: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for i in range(10)}
        for (predicted_class, actual_class) in results:
            # True Positive (TP) and True Negative (TN)
            if predicted_class == actual_class:
                confusion_matrix[actual_class]["TP"] += 1
                for i in range(10):
                    if i != actual_class:
                        confusion_matrix[i]["TN"] += 1
            # False Positive (FP) and False Negative (FN)
            else:
                confusion_matrix[actual_class]["FN"] += 1
                confusion_matrix[predicted_class]["FP"] += 1
                for i in range(10):
                    if i != actual_class and i != predicted_class:
                        confusion_matrix[i]["TN"] += 1
        matrix = np.zeros((10, 4), dtype=int)
        for _, (class_label, metrics) in enumerate(confusion_matrix.items()):
            matrix[9-class_label, 0] = metrics['TP']
            matrix[9-class_label, 1] = metrics['FP']
            matrix[9-class_label, 2] = metrics['TN']
            matrix[9-class_label, 3] = metrics['FN']
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto', extent=[-0.5, 3.5, -0.5, 9.5])
        plt.title('Confusion Matrix')
        plt.colorbar()
        classes = [f'{i}' for i in range(10)]
        tick_marks = np.arange(len(classes))
        plt.xticks(np.arange(4), ['True Positive', 'False Positive', 'True Negative', 'False Negative'])
        plt.yticks(tick_marks, classes)
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        # Displaying the values in the matrix
        for i in range(10):
            for j in range(4):
                plt.text(j, i, str(matrix[i, j]), ha='center', va='center', color='black')

        plt.show()
        return confusion_matrix


def activation_fun(z, type):
    if type == 'sigmoid':
        return sigmoid(z)
    elif type == 'tanh':
        return tanh(z)
    elif type == 'ReLU':
        return relu(z)
    else:
        return leaky_relu(z)


def activation_prime(z, type):
    if type == 'sigmoid':
        return sigmoid_prime(z)
    elif type == 'tanh':
        return tanh_prime(z)
    elif type == 'ReLU':
        return relu_prime(z)
    else:
        return leaky_relu_prime(z)


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def relu(z):
    return np.maximum(0, z)


def relu_prime(z):
    return np.where(z > 0, 1, 0)


def leaky_relu(z):
    return np.maximum(0.1*z, z)


def leaky_relu_prime(z):
    return np.where(z > 0, 1, 0.1)


def tanh(z):
    return np.tanh(z)


def tanh_prime(z):
    return 1-np.tanh(z)**2
