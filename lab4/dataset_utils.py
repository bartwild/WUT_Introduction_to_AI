import random


def split_random_to_train_and_test_data(attrs_vals, class_vals, percent_of_train_data):
    """
    Splits the given dataset into training and testing data randomly.

    Args:
        attrs_vals (list): List of attribute values.
        class_vals (list): List of class values.
        percent_of_train_data (float): Percentage of data to be used for training.

    Returns:
        tuple: A tuple containing the training data and testing data.
            The training data is a tuple with two elements:
                - A dictionary with the keys "attrs_index" and "attrs_vals".
                - A list of class values.
            The testing data is a tuple with two elements:
                - A dictionary with the keys "attrs_index" and "attrs_vals".
                - A list of class values.
    """
    if percent_of_train_data <= 0 or percent_of_train_data >= 100 or len(attrs_vals) == 0:
        return
    attrs_indexes = [i for i in range(len(attrs_vals[0]))]
    train_data = ({"attrs_index": attrs_indexes, "attrs_vals": []}, [])
    test_data = ({"attrs_index": attrs_indexes, "attrs_vals": []}, [])
    attrs_vals_copy = list(attrs_vals)
    class_vals_copy = list(class_vals)
    train_size = int(len(attrs_vals) * (percent_of_train_data/100))

    while len(train_data[0]["attrs_vals"]) < train_size:
        index = random.randrange(len(attrs_vals_copy))
        train_data[0]["attrs_vals"].append(attrs_vals_copy.pop(index))
        train_data[1].append(class_vals_copy.pop(index))
    while len(attrs_vals_copy) > 0:
        index = random.randrange(len(attrs_vals_copy))
        test_data[0]["attrs_vals"].append(attrs_vals_copy.pop(index))
        test_data[1].append(class_vals_copy.pop(index))
    combined_train_data = list(zip(train_data[0]["attrs_vals"], train_data[1]))
    random.shuffle(combined_train_data)
    train_data[0]["attrs_vals"][:], train_data[1][:] = zip(*combined_train_data)
    return train_data, test_data


def split_to_sorted_train_and_test_data(attrs_vals, class_vals, percent_of_train_data):
    """
    Splits the given dataset into training and testing data and sort test_data

    Args:
        attrs_vals (list): List of attribute values.
        class_vals (list): List of class values.
        percent_of_train_data (float): Percentage of data to be used for training.

    Returns:
        tuple: A tuple containing the training data and testing data.
            The training data is a tuple with two elements:
                - A dictionary with the keys "attrs_index" and "attrs_vals".
                - A list of class values.
            The testing data is a tuple with two elements:
                - A dictionary with the keys "attrs_index" and "attrs_vals".
                - A list of class values.
    """
    if percent_of_train_data <= 0 or percent_of_train_data >= 100 or len(attrs_vals) == 0:
        return
    attrs_indexes = [i for i in range(len(attrs_vals[0]))]
    train_data = ({"attrs_index": attrs_indexes, "attrs_vals": []}, [])
    test_data = ({"attrs_index": attrs_indexes, "attrs_vals": []}, [])
    attrs_vals_copy = list(attrs_vals)
    class_vals_copy = list(class_vals)
    train_size = int(len(attrs_vals) * (percent_of_train_data/100))

    while len(train_data[0]["attrs_vals"]) < train_size:
        index = random.randrange(len(attrs_vals_copy))
        train_data[0]["attrs_vals"].append(attrs_vals_copy.pop(index))
        train_data[1].append(class_vals_copy.pop(index))
    while len(attrs_vals_copy) > 0:
        index = random.randrange(len(attrs_vals_copy))
        test_data[0]["attrs_vals"].append(attrs_vals_copy.pop(index))
        test_data[1].append(class_vals_copy.pop(index))
    sorted_train_data = sorted(zip(train_data[0]["attrs_vals"], train_data[1]), key=lambda x: x[1][-1])
    train_data = ({"attrs_index": attrs_indexes, "attrs_vals": [item[0] for item in sorted_train_data]}, [item[1] for item in sorted_train_data])
    return train_data, test_data


def get_data(name):
    """
    Reads a file and extracts attribute values and class labels.

    Args:
        name (str): The name of the file to be read.

    Returns:
        tuple: A tuple containing two lists. The first list contains attribute values for each row,
               and the second list contains class labels for each row.
    """
    row_attrs_vals = []
    row_class_vals = []

    file = open(name)

    for _, line in enumerate(file):
        vals = line.replace("\n", "").replace(" ", "").split(",")
        if len(vals) < 2:
            continue
        row_attrs_vals.append(vals[:-1])
        row_class_vals.append(vals[-1])
    return row_attrs_vals, row_class_vals
