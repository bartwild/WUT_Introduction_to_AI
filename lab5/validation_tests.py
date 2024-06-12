from matplotlib import pyplot as plt


def best_epochs(net, train, val, eps, bs, lr, lbd):
    """
    Finds the best number of epochs for a neural network model based on the given parameters.

    Args:
        net (NeuralNetwork): The neural network model.
        train (list): The training data.
        val (list): The validation data.
        eps (list): The list of epochs to test.
        bs (int): The batch size.
        lr (float): The learning rate.
        lbd (float): The lambda value for regularization.

    Returns:
        tuple: A tuple containing the best number of epochs and the list of accuracies for each epoch.
    """
    accuracies = []
    for ep in eps:
        print(f"Epochs: {ep}")
        accuracies.append(net.stochastic_gradient_descent(
            training_data=train,
            test_data=val,
            epochs=ep,
            batch_size=bs,
            learn_rate=lr,
            lambda_=lbd))
    best = eps[accuracies.index(max(accuracies))]
    return best, accuracies


def best_batch_size(net, train, val, ep, bss, lr, lbd):
    """
    Finds the best batch size for a neural network model based on the given parameters.

    Args:
        net (NeuralNetwork): The neural network model.
        train (list): The training data.
        val (list): The validation data.
        ep (int): The number of epochs.
        bss (list): The list of batch sizes to test.
        lr (float): The learning rate.
        lbd (float): The lambda value for regularization.

    Returns:
        tuple: A tuple containing the best batch size and a list of accuracies for each batch size.
    """
    accuracies = []
    for bs in bss:
        print(f"Batch size: {bs}")
        accuracies.append(net.stochastic_gradient_descent(
            training_data=train,
            test_data=val,
            epochs=ep,
            batch_size=bs,
            learn_rate=lr,
            lambda_=lbd))
    best = bss[accuracies.index(max(accuracies))]
    return best, accuracies


def best_learnrate(net, train, val, ep, bs, lrs, lbd):
    """
    Finds the best learning rate for a neural network model.

    Args:
        net (NeuralNetwork): The neural network model.
        train (list): The training data.
        val (list): The validation data.
        ep (int): The number of epochs.
        bs (int): The batch size.
        lrs (list): The list of learning rates to test.
        lbd (float): The lambda value for regularization.

    Returns:
        tuple: A tuple containing the best learning rate and a list of accuracies for each learning rate.
    """
    accuracies = []
    for lr in lrs:
        print(f"Learn rate: {lr}")
        accuracies.append(net.stochastic_gradient_descent(
            training_data=train,
            test_data=val,
            epochs=ep,
            batch_size=bs,
            learn_rate=lr,
            lambda_=lbd))
    best = lrs[accuracies.index(max(accuracies))]
    return best, accuracies


def best_lambda(net, train, val, ep, bs, lr, lbds):
    """
    Finds the best lambda value for a neural network using stochastic gradient descent.

    Args:
        net (NeuralNetwork): The neural network model.
        train (list): The training data.
        val (list): The validation data.
        ep (int): The number of epochs.
        bs (int): The batch size.
        lr (float): The learning rate.
        lbds (list): The list of lambda values to test.

    Returns:
        tuple: The best lambda value and the list of accuracies for each lambda value.
    """
    accuracies = []
    for lbd in lbds:
        print(f"Lambda: {lbd}")
        accuracies.append(net.stochastic_gradient_descent(
            training_data=train,
            test_data=val,
            epochs=ep,
            batch_size=bs,
            learn_rate=lr,
            lambda_=lbd))
    best = lbds[accuracies.index(max(accuracies))]
    return best, accuracies


def plot(tested_val, values, accuracies):
    """
    Plots the accuracy graph based on the tested value, values, and accuracies.

    Parameters:
    tested_val (str): The tested value.
    values (list): The list of values.
    accuracies (list): The list of accuracies.

    Returns:
    None
    """
    plt.plot(values, accuracies)
    plt.xlabel(tested_val)
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy - No. of {tested_val} graph.")
    # plt.savefig(f"Accuracy - No. of {tested_val} graph.")
    plt.show()
