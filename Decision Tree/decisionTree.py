"""
Filename : decisionTree.py

Author : Archit Joshi

Description : Demonstration of decision tree learning algorithm from scratch.
Uses the Breast Cancer Wisconsin (Diagnostic) dataset available from the
UC Irvine Machine Learning Repository. More information available in the
DTreeREADME.md file.

Scikit-learn used only for data splitting purposes.

Citation:
Wolberg,William, Mangasarian,Olvi, Street,Nick, and Street,W.. (1995).
Breast Cancer Wisconsin (Diagnostic).
UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.

Language : python3
"""
import os

import pandas as pd
import numpy as np
import math as m
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from ucimlrepo import fetch_ucirepo
from graphviz import Source
import pickle


class Node:
    """
    Class to represent a Node in a tree.
    """

    def __init__(self, attribute=None, threshold=None, left=None, right=None,
                 class_label=None, is_left=None, depth=None):
        self.attribute = attribute
        self.threshold = threshold
        self.left = left
        self.right = right
        self.class_label = class_label
        self.is_left = is_left
        self.depth = depth

    def __str__(self, depth=0):
        indent = "    " * depth
        if self.class_label is not None:
            return f'{indent}Leaf Node: Class {self.class_label} (Depth: {depth}, {"Left" if self.is_left else "Right"} node)'
        else:
            branch = "Left" if self.is_left else "Right"
            return f'{indent}Node: {self.attribute} <= {self.threshold} ({branch} branch, Depth: {depth})\n' + \
                f'{self.left.__str__(depth + 1)}\n' + \
                f'{self.right.__str__(depth + 1)}'


def parseDataSet():
    """
    Function to parse the dataset from the Data set repository online.
    DEPENDENCY : Install the ucimlrepo package -> pip install ucimlrepo

    :return: dataset as a pandas dataframe
    """
    # fetch dataset
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    # data (as pandas dataframes)
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets

    dataframe = pd.concat([X, y], axis=1)
    # Write to csv for easy access later
    dataframe.to_csv(os.getcwd() + '//dataset.csv', index=False)
    print(dataframe)
    columns = dataframe.columns.tolist()
    return X, y, dataframe


def splitData(X, y):
    """
    Helper function to split data into training and testing dataset.
    We will use sklearn library just for this portion to get a random split.

    :param X: Features dataframe from the dataset
    :param y: Labels dataframe from the dataset
    :return: training, test and validation split
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    data_training = pd.concat([X_train, y_train], axis=1)
    data_testing = pd.concat([X_test, y_test], axis=1)

    return data_training, data_testing


def stoppingCriteria(data, depth):
    """
    Helper function to check stopping criteria for the base call of the
    recursive decision tree function.

    topping Criteria -
    a. There are less than 15 data points in a node, OR
    b. Depth is over 20.

    :param data: data set for this node
    :param depth: depth of this node
    :return: None
    """
    # Stopping criteria (a) and (c)
    if len(data) <= 25 or depth > 20:
        return True

    class_counts = pd.value_counts(data['Diagnosis'])
    percentages = class_counts / len(data)
    if max(percentages) >= 0.9:
        return True
    # Keep recursing
    return False


def majorityClass(data):
    """
    Helper function to determine the majority class for the data.

    :param data: training data with labels
    :return: majority label
    """
    # Use pandas value_counts() function to return a count for each label.
    value_counts = data['Diagnosis'].value_counts()
    majority_value = value_counts.idxmax()
    return majority_value


def calculateEntropy(data):
    """
    Calculate entropy of the node.

    :param data: training data
    :return: entropy for the data
    """
    classes, counts = np.unique(data['Diagnosis'], return_counts=True)
    probablities = counts / len(data)
    return -np.sum(probablities * np.log2(probablities))


def calculateWeightedEntropy(data, feature, split_val):
    """
    Calculate weighted entropy for the child nodes using the best split point.

    :param data: training data
    :param feature: feature to split data on
    :param split_val: best split point (threshold)
    :return: weighted entropy
    """
    # Divide into subsets based on split point
    subset_A = data[data[feature] <= split_val]
    subset_B = data[data[feature] > split_val]

    if len(subset_A) == 0 or len(subset_B) == 0:
        return 0

    # Calculate entropy of the nodes
    entropy_A = calculateEntropy(subset_A)
    entropy_B = calculateEntropy(subset_B)

    weightedEntropy = ((len(subset_A) / len(data)) * entropy_A) + (
            (len(subset_B) / len(data)) * entropy_B)

    return weightedEntropy


def calculateSplitInfo(data, feature, split_val):
    """
    Calculate split info required for calculating information gain ratio.

    :param data: training data
    :param feature: feature chosen for the split
    :param split_val: best split point (threshold)
    :return: split info for this feature
    """
    # Calculate subsets based on split point
    subset_A = data[data[feature] <= split_val]
    subset_B = data[data[feature] > split_val]

    prob_A = len(subset_A) / len(data)
    prob_B = len(subset_B) / len(data)

    # Use math library for calculating logs
    term_A = prob_A * m.log2(prob_A) if prob_A != 0 else 0
    term_B = prob_B * m.log2(prob_B) if prob_B != 0 else 0

    return -(term_A + term_B)


def calculateInfoGainRatio(data, feature):
    """
    Wrapper function to calculate information gain ratio for the attribute.

    Objective function : Maximize information gain ratio

    :param data: training data with labels
    :param feature: feature chosen to test
    :return: highest information gain ratio
    """
    best_information_gain_ratio = float("-inf")
    best_threshold = None

    # Generate a list of possible thresholds based on the unique values for the
    # feature
    # ROOM FOR IMPROVEMENT - Check other methods to determine possible split
    # points, binning for larger datasets to reduce overload
    thresholds = np.unique(data[feature])

    for threshold in thresholds:
        # Calculate entropy and information gain
        entropy = calculateEntropy(data)
        weighted_entropy = calculateWeightedEntropy(data, feature, threshold)
        information_gain = entropy - weighted_entropy

        # Calculate split information
        split_info = calculateSplitInfo(data, feature, threshold)

        # Calculate information gain ratio
        try:
            information_gain_ratio = information_gain / split_info
        except ZeroDivisionError as e:
            information_gain_ratio = 0

        # Maximize objective function and set best values accordingly
        if information_gain_ratio > best_information_gain_ratio:
            best_information_gain_ratio = information_gain_ratio
            best_threshold = threshold

    return best_information_gain_ratio, best_threshold


def getBestSplitPoint(data):
    """
    Wrapper function that will calculate the information gain for each attribute
    and determine the best attribute to split the node on with the appropriate
    threshold.

    :param data: training data with labels
    :return: best feature, best threshold
    """
    attributes = data.columns.tolist()[:-1]

    # Dictionary that stores Attribute -> Information Gain Ratio, Threshold pairs
    infoGainRatio = dict()

    # Get the best threshold and information gain for each attribute
    for attribute in attributes:
        infoGainRatio[attribute] = calculateInfoGainRatio(data, attribute)

    # Best attribute is the one with maximum Information gain ratio
    best_attribute = max(infoGainRatio, key=lambda k: infoGainRatio[k][0])
    best_threshold = infoGainRatio[best_attribute][1]

    return best_attribute, best_threshold


def buildDecisionTree(data, depth, is_left):
    """
    Recursively build decision tree using the training data and use pickle to
    store the model.

    :param data: training data with labels.
    :param depth: current depth of the tree.
    :param is_left: denote if left branch.
    :return: decision tree model
    """

    # Base Case returning leaf node
    if stoppingCriteria(data, depth):
        return Node(class_label=majorityClass(data), depth=depth,
                    is_left=is_left)

    # Calculate best attribute and split point
    attribute, threshold = getBestSplitPoint(data)

    # Make left and right subsets
    left_data = data[data[attribute] <= threshold]
    right_data = data[data[attribute] > threshold]

    # Edge cases to return leaf nodes
    if left_data is None:
        Node(class_label=majorityClass(right_data), depth=depth,
             is_left=is_left)
    elif right_data is None:
        Node(class_label=majorityClass(left_data), depth=depth,
             is_left=is_left)

    # Recursively create left and right subtree
    left_subtree = buildDecisionTree(left_data, depth + 1, is_left=True)
    right_subtree = buildDecisionTree(right_data, depth + 1, is_left=False)

    # Returning parent node
    return Node(attribute=attribute, threshold=threshold,
                left=left_subtree, right=right_subtree, is_left=is_left,
                depth=depth)


def main():
    features, labels, data = parseDataSet()
    training_data, testing_data = splitData(features, labels)

    # Build decision tree model recursively using training data
    decisionTree = buildDecisionTree(training_data, depth=0, is_left=None)

    print(decisionTree)


if __name__ == "__main__":
    main()
