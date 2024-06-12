import seaborn as sn
from matplotlib import pyplot as plt
import random
import pandas as pd


def metrics_vector(train_data, test_data, predict_func):
    """
    Calculates various metrics for evaluating the performance of a classifier.

    Parameters:
    train_data (DataFrame): The training dataset.
    test_data (DataFrame): The test dataset.

    Returns:
    list: A list containing the following metrics in order: recall, fall_out, precision, accuracy, f1_score.
    """
    random.seed(420)
    predictions, real_values = predict_func(train_data, test_data)
    real_values = test_data[test_data.columns[0]].values
    t_p, f_p, f_n, t_n = 0, 0, 0, test_data.shape[0]
    for score_value in test_data[test_data.columns[0]].unique():
        for prediction, real_value in zip(predictions, real_values):
            if prediction == score_value and real_value != score_value:
                f_p += 1
            if real_value == score_value and prediction != score_value:
                f_n += 1
            if real_value == score_value and prediction == score_value:
                t_p += 1
    f_p /= 2
    f_n /= 2
    t_n = t_n - t_p - f_n - f_p
    if (t_p + f_n) == 0:
        recall = 0
    else:
        recall = t_p / (t_p + f_n)
    if (f_p+t_p == 0):
        fall_out = 0
    else:
        fall_out = f_p / (f_p + t_p)
    try:
        precision = t_p / (t_p + f_p)
    except ZeroDivisionError:
        precision = 0
    accuracy = (t_p + t_n) / (t_p + t_n + f_n + f_p)
    if precision == 0:
        f1_score = 0
    else:
        f1_score = (2 * recall * precision) / (recall + precision)
    return [recall, fall_out, precision, accuracy, f1_score]


def confusion_matrix(predictions, real_values, name):
    """
    Generate a confusion matrix plot based on the predicted and actual values.

    Parameters:
    predictions (list): List of predicted values.
    real_values (list): List of actual values.
    name (str): Name of the output file.

    Returns:
    None
    """
    data = {'y_Actual':    real_values,
            'y_Predicted': predictions}
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Predicted'], df['y_Actual'], rownames=['Predicted'], colnames=['Actual'])
    sn.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g')
    plt.savefig(f"{name}.png")


def testing_data(dataset):
    """
    Plot scatter plots for each pair of features in the dataset.

    Args:
        dataset (pandas.DataFrame): The dataset containing the features.

    Returns:
        None
    """
    minisets, values = split_by_class(dataset)
    category_list = dataset.columns.tolist()
    del (category_list[0])
    del (category_list[11])
    for i in range(len(category_list)):
        for j in range(len(category_list)):
            if i < j:
                cat_1 = category_list[i]
                cat_2 = category_list[j]
                for k in range(len(values)):
                    cat_1 = category_list[i]
                    cat_2 = category_list[j]
                    list_1 = minisets[k][cat_1].tolist()
                    list_2 = minisets[k][cat_2].tolist()
                    color = 'seagreen'
                    if values[k] == 1:
                        color = 'royalblue'
                    if values[k] == 2:
                        color = 'deeppink'
                    plt.plot(list_1, list_2, 'o', color=color, label=f'Class {values[k]}')
                plt.legend()
                title = cat_1 + " - " + cat_2
                plt.title(title)
                plt.xlabel(cat_1)
                plt.ylabel(cat_2)
                title_png = title + ".png"
                plt.savefig(title_png)
                plt.clf()


def split_by_class(dataset):
    """
    Splits the dataset into multiple mini datasets based on the unique values of the target class.

    Parameters:
    dataset (pandas.DataFrame): The dataset to be split.

    Returns:
    mini_datasets (list): A list of mini datasets, each containing the rows corresponding to a unique value of the target class.
    values (list): A list of unique values of the target class.
    """
    mini_datasets = []
    values = []
    classes = dataset.columns.tolist()[0]
    for value in dataset[classes].unique():
        values.append(value)
        mini_class = dataset[dataset[classes] == value]
        mini_class.drop(classes, axis=1, inplace=True)
        mini_datasets.append(mini_class)
    return mini_datasets, values
