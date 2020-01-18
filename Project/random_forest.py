import pandas as pd
from Project.decision_tree import DecisionTree
from statistics import mode
from statistics import StatisticsError


class RandomForest:
    def __init__(self, n_trees, n_features, max_depth, min_samples_leaf):
        self.trees = n_trees
        self.max_depth = max_depth
        self.n_features = n_features
        self.min_size = min_samples_leaf
        self.tree_list = list()

    def fit(self, X, y):
        for i in range(self.trees):
            sample = X.sample(self.n_features, axis=1)
            data = sample.join(y)
            dt = DecisionTree()
            dt.features = sample.columns.values.tolist()
            dt.build_tree(data.values, self.max_depth, self.min_size)
            self.tree_list.append(dt)

    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    def predict(self, X, y):
        data = X.join(y)
        predictions = list()
        for row in data.values:
            outcomes = list()
            for tree in self.tree_list:
                tree_columns_index = [data.columns.get_loc(f) for f in tree.features]
                prediction = tree.predict(tree.tree, [row[i] for i in tree_columns_index])
                outcomes.append(prediction)
            try:
                mfv = mode(outcomes)
            except StatisticsError:
                mfv = outcomes[0]
            # print(f'Expected={row[-1]}, Predicted={mfv}')
            predictions.append(mfv)
        print(f"Accuracy: {self.accuracy_metric(data.iloc[:, -1].values, predictions)}")
        return predictions
