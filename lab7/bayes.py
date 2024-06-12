import pandas as pd
import statistics
import math
import sklearn.model_selection as sc
from sklearn.utils import shuffle
from analysis_utils import metrics_vector, confusion_matrix, testing_data, split_by_class


def get_dataframe(file_name="wine.data"):
    """
    Reads a CSV file and returns a pandas DataFrame.

    Parameters:
    - file_name (str): The name of the CSV file to be read. Default is "wine.data".
    - header_list (list): A list of column names for the DataFrame. Default is a list of column names for wine data.

    Returns:
    - pandas.DataFrame: The DataFrame containing the data from the CSV file.
    """
    return pd.read_csv(file_name)


def stats_for_column(column):
    """
    Calculate the mean, standard deviation, and length of a given column.

    Args:
        column (list): A list of numerical values.

    Returns:
        list: A list containing the mean, standard deviation, and length of the column.
    """
    return [statistics.mean(column), statistics.stdev(column), len(column)]


def stats_for_dataset(dataset):
    """
    Calculate statistics for each column in the dataset.

    Args:
        dataset (pandas.DataFrame): The dataset to calculate statistics for.

    Returns:
        list: A list of statistics for each column in the dataset.
    """
    stats_for_dataset = []
    for column in dataset.columns.tolist():
        stats_for_dataset.append(stats_for_column(dataset[column].tolist()))
    return stats_for_dataset


def stats_for_classes(dataset):
    """
    Calculate statistics for each class in the dataset.

    Args:
        dataset (list): The dataset containing the samples.

    Returns:
        tuple: A tuple containing the statistics for each class and the unique class values.
    """
    mini_classes, values = split_by_class(dataset)
    classes_stats = []
    for mini_class in mini_classes:
        classes_stats.append(stats_for_dataset(mini_class))
    return classes_stats, values


def gauss_func(x, average, stdev):
    return 1/math.sqrt(2*math.pi*stdev**2) * math.e**(-(x - average)**2/(2*stdev**2))


def predicted_probabilities(stats, row):
    """
    Calculates the predicted probabilities for each class based on the given statistics and input row.

    Parameters:
    stats (list): The statistics for each class, including mean and standard deviation.
    row (list): The input row for which the probabilities are calculated.

    Returns:
    list: The predicted probabilities for each class.
    """
    rows = sum([element[0][2] for element in stats])
    probs = []
    for class_stats in stats:
        probab = class_stats[0][2] / rows
        for i in range(len(class_stats)):
            probab *= gauss_func(row[i], class_stats[i][0], class_stats[i][1])
        probs.append(probab)
    return probs


def predict_class(stats, values, row):
    """
    Predicts the class for a given row based on the statistics and values.

    Parameters:
    stats (list): The statistics used for prediction.
    values (list): The corresponding class values.
    row (list): The row for which the class is to be predicted.

    Returns:
    The predicted class for the given row.
    """
    probs = predicted_probabilities(stats, row)
    class_prob = max(probs)
    max_index = probs.index(class_prob)
    return values[max_index]


def predict_dataset(train_data, test_data):
    """
    Predicts the class labels for a given test dataset using the Naive Bayes classifier.

    Parameters:
    train_data (DataFrame): The training dataset used to train the Naive Bayes classifier.
    test_data (DataFrame): The test dataset for which class labels are to be predicted.

    Returns:
    predictions (list): The predicted class labels for the test dataset.
    real_values (list): The actual class labels for the test dataset.
    """
    classes_stats, values = stats_for_classes(train_data)
    predictions = []
    real_values = []
    for _, row in test_data.iterrows():
        chosen_row = row.tolist()
        if len(chosen_row) > 0:
            row_values = chosen_row[1:]
            real_value = chosen_row[0]
            predictions.append(predict_class(classes_stats, values, row_values))
            real_values.append(int(real_value))
    return predictions, real_values


def main():
    dataset = get_dataframe()
    # testing_data(dataset)
    # dataset = shuffle(dataset)
    train_data = dataset[0:int(len(dataset)*0.8)]
    test_data = dataset[int(len(dataset)*0.2):]
    # train_data, test_data = sc.train_test_split(dataset, test_size=0.5)
    predictions, real_values = predict_dataset(train_data, test_data)
    confusion_matrix(predictions, real_values, "CM")
    testing_data(dataset)
    print(metrics_vector(train_data, test_data, predict_dataset))
    for i in range(len(predictions)):
        if predictions[i] != real_values[i]:
            print(f'Predicted: {predictions[i]}, Real: {real_values[i]}')


if __name__ == "__main__":
    main()
