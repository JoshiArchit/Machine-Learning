"""
Filename : linearclassifier.py
Author : Archit Joshi
Description : Implementing linear classifier
numpy and matplotlib.
Language : python3
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib

SEED = 42


def drawGraph(data):
    # draw graph
    pass


def irisDataSet():
    """
    Function to create a stacked table of iris data set

    :return: dataset with features and labels in a single table
    """
    features, labels = load_iris(return_X_y=True, as_frame=False)
    data_table = np.hstack((features, labels.reshape(-1, 1)))
    return data_table


def trainLinearClassifier(dataset):
    """
    Dataset with only features to train linear classifier.

    :param dataset: features without labels
    :return: classifier
    """
    print("Inside Classifier")
    print(dataset)


def main():
    # Data stack needs to be created with [features, labels]
    data = irisDataSet()
    features, labels = data[:, :-1], data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                        test_size=0.2,
                                                        random_state=SEED)

    # Train classifier



if __name__ == "__main__":
    main()
