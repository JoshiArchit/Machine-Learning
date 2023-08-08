"""
Filename : linearclassifier.py
Author : Archit Joshi
Description : Implementing linear classifier
numpy and matplotlib.
Language : python3
"""
import numpy as np
from sklearn.datasets import load_iris


def irisDataSet():
    """
    Function to create a stacked table of iris data set

    :return: dataset with features and labels in a single table
    """
    features, labels = load_iris(return_X_y=True, as_frame=False)
    data_table = np.hstack((features, labels.reshape(-1, 1)))
    return data_table


def main():
    data = irisDataSet()


if __name__ == "__main__":
    main()
