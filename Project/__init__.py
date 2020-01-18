import pandas as pd
import numpy as np
from Project.decision_tree import DecisionTree
from Project.random_forest import RandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
import math
import time

DEPTH = 2
MIN_SAMPLES = 7


def import_data():
    df = pd.read_csv('SAHeart.data', sep=',', header=0, index_col=0)
    df.head()

    # One hot encoding
    cat_columns = df.select_dtypes(['object']).columns
    df[cat_columns] = df[cat_columns].astype('category')
    print(cat_columns)
    one_hot = pd.get_dummies(df[cat_columns], drop_first=False)
    df = df.drop(df[cat_columns].columns.values, axis=1)
    df = df.join(one_hot)
    return df


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def add_data(index, svc, test_data, max_depth, min_samples, results, data, time):
    y_pred = svc.predict(test_data, results)
    precision, recall, fscore, support = score(results, y_pred)

    print('precision: {}'.format(np.mean(precision)))
    print('recall: {}'.format(np.mean(recall)))
    print('fscore: {}'.format(np.mean(fscore)))
    print('support: {}'.format(support))
    data.append([index, max_depth, min_samples, time, np.mean(precision), np.mean(fscore), np.mean(recall)])


def dt_test(df):
    dataset = df.drop(['chd'], axis=1)
    dataset = dataset.join(df['chd'])

    training_data, test_data = train_test_split(dataset, test_size=0.2, random_state=10)
    dt = DecisionTree()
    tree = dt.build_tree(training_data.values, DEPTH, MIN_SAMPLES)
    predicted = list()
    for row in test_data.values:
        prediction = dt.predict(tree, row)
        predicted.append(prediction)
    print(f"Accuracy: {accuracy_metric(test_data.iloc[:, -1].values, predicted)}")


if __name__ == '__main__':
    data = import_data()
    print(f"Max depth: {DEPTH}")
    print(f"Min samples: {MIN_SAMPLES}")
    # print("Decision Tree")
    # dt_test(data)
    resultsColumns = ["index", "max_depth", "min_samples", "time", "accuracy", "f1_score", "recall"]
    result = []

    n_features = int(round(math.sqrt(len(data.columns))))
    print("Random Forest TEST SET")
    y = data.chd
    X = data.drop(['chd'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    '''
    index = 0
    for dpth in range(2, 10):
        for min_smpl in range(2, 10):
            start = time.time()
            RF = RandomForest(n_trees=100, n_features=n_features, max_depth=dpth, min_samples_leaf=min_smpl)
            RF.fit(X_train, y_train)
            end = time.time()
            add_data(index, RF, X_test, dpth, min_smpl, y_test, result, end - start)
            index += 1

    results_df = pd.DataFrame(result, columns=resultsColumns)
    results_df.to_csv("SAHeart_MYRF100.csv", index=False, header=True)
    '''
    rf = RandomForest(n_trees=100, n_features=n_features, max_depth=DEPTH, min_samples_leaf=MIN_SAMPLES)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test, y_test)

    # print("Random Forest TRAIN SET")
    # predictions = rf.predict(X_train, y_train)
