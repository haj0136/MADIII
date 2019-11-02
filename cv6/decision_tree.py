import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def import_data():
    data = pd.read_csv('sep.csv', sep=';', header=None)

    # Printing the dataset shape
    print("Dataset Length: ", len(data))
    print("Dataset Shape: ", data.shape)

    print("Dataset: ", data.head())
    return data


def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini


def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    # right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


if __name__ == '__main__':
    dataset = import_data()
    test = dataset.iloc[:, 0]
    dic = {-1: 'r', 1: 'b'}
    colors = [dic[value] for value in list(dataset.iloc[:, 2].values)]
    plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c=colors)
    # plt.show()
    training_data, test_data = train_test_split(dataset, test_size=0.2)
    """
    split = get_split(dataset.values)
    print('Split: [X%d < %.3f]' % ((split['index'] + 1), split['value']))
    """
    tree = build_tree(training_data.values, 10, 5)
    predicted = list()
    for row in test_data.values:
        prediction = predict(tree, row)
        predicted.append(prediction)
        print(f'Expected={row[-1]}, Predicted={prediction}')
        if prediction == row[-1]:
            plt.scatter(row[0], row[1], c="g")
        else:
            plt.scatter(row[0], row[1], c="y")
    print(f"Accuracy: {accuracy_metric(test_data.iloc[:, -1].values, predicted)}")
    plt.show()
    print()
