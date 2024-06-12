from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from keras.datasets import mnist


def load_data(image_size):
    """
    Load and preprocess the digit dataset.

    Args:
        image_size (int): The size of the digit images.

    Returns:
        tuple: A tuple containing the preprocessed training data, validation data, and test data.
            The training data is a list of tuples, where each tuple contains a reshaped image and its corresponding vectorized label.
            The validation data is a list of tuples, where each tuple contains a reshaped image and its corresponding label.
            The test data is a list of tuples, where each tuple contains a reshaped image and its corresponding label.
    """
    digits = datasets.load_digits()

    tuple_data = [data for data in zip(digits.data, digits.target)]

    train, other = train_test_split(tuple_data, train_size=0.7)
    validation, test = train_test_split(other, test_size=0.5)

    train_data = [(np.reshape(image, (image_size, 1)), vectorized_result(digit)) for image, digit in train]
    validation_data = [(np.reshape(image, (image_size, 1)), digit) for image, digit in validation]
    test_data = [(np.reshape(image, (image_size, 1)), digit) for image, digit in test]
    print(len(train_data), len(validation_data), len(test_data))
    return (train_data, validation_data, test_data)


def vectorized_result(result_digit):
    """
    Convert a digit into a vectorized representation.

    Args:
        result_digit (int): The digit to be converted.

    Returns:
        numpy.ndarray: A vectorized representation of the digit.
    """
    vector = np.zeros((10, 1))
    vector[result_digit] = 1.0
    return vector


def load_data2(image_size):
    """
    Load and preprocess the MNIST dataset.

    Args:
        image_size (int): The size of the digit images.

    Returns:
        tuple: A tuple containing the preprocessed training data, validation data, and test data.
            The training data is a list of tuples, where each tuple contains a reshaped image and its corresponding vectorized label.
            The validation data is a list of tuples, where each tuple contains a reshaped image and its corresponding label.
            The test data is a list of tuples, where each tuple contains a reshaped image and its corresponding label.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train.reshape((len(x_train), image_size))
    x_test = x_test.reshape((len(x_test), image_size))

    train_data = list(zip(x_train, y_train))
    test_data = list(zip(x_test, y_test))

    train_data = [(np.reshape(image, (image_size, 1)), vectorized_result(digit)) for image, digit in train_data]
    test_data = [(np.reshape(image, (image_size, 1)), digit) for image, digit in test_data]
    print(len(train_data), len(test_data))
    return (train_data, test_data)
