import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import tree
import time
import math


def print_report(svc, test_data, label, results):
    y_pred = svc.predict(test_data)
    print(label)
    print(classification_report(results, y_pred))


def add_data(index, svc, test_data, max_depth, min_samples, results, data, time):
    y_pred = svc.predict(test_data)
    precision, recall, fscore, support = score(results, y_pred)

    print('precision: {}'.format(np.mean(precision)))
    print('recall: {}'.format(np.mean(recall)))
    print('fscore: {}'.format(np.mean(fscore)))
    print('support: {}'.format(support))
    data.append([index, max_depth, min_samples, time, np.mean(precision), np.mean(fscore), np.mean(recall)])


if __name__ == '__main__':
    data = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    data = data.drop('proto', axis=1)
    resultsColumns = ["index", "max_depth", "min_samples", "time", "accuracy", "f1_score", "recall"]
    data.head()
    result = []

    # One hot encoding
    cat_columns = data.select_dtypes(['object']).columns
    data[cat_columns] = data[cat_columns].astype('category')
    print(cat_columns)
    one_hot = pd.get_dummies(data[cat_columns], drop_first=False)
    data = data.drop(data[cat_columns].columns.values, axis=1)
    data = data.join(one_hot)

    X = data.drop(['label'], axis=1).values
    y = data.label.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    n_features = int(round(math.sqrt(len(data.columns))))

    index = 0
    for dpth in range(2, 10):
        for min_smpl in range(2, 10):
            start = time.time()
            RF = RandomForestClassifier(n_estimators=100, max_depth=dpth, random_state=0, max_features=n_features,
                                        min_samples_leaf=min_smpl)
            RF.fit(X_train, y_train)
            end = time.time()
            add_data(index, RF, X_test, dpth, min_smpl, y_test, result, end - start)
            index += 1

    results_df = pd.DataFrame(result, columns=resultsColumns)
    print("Complete")
    results_df.to_csv("UNSW_RF100.csv", index=False, header=True)
    # clf = tree.DecisionTreeClassifier(max_depth=4, random_state=0, min_samples_leaf=4)
    # clf.fit(X_train, y_train)
    # print("DT TEST SET")
    # print_report(clf, X_test, "Decision Tree", y_test)
    # print("RF TEST SET")
    # print_report(RF, X_test, "Random Forest 100", y_test)
    # print("RF TRAIN SET")
    # print_report(RF, X_train, "Random Forest 100", y_train)



