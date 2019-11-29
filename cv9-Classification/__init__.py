import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support as score
import time


def print_report(svc, test_data, label, results):
    y_pred = svc.predict(test_data)
    print(label)
    print(classification_report(results, y_pred))


def add_data(index, svc, test_data, label, dataset_label, results, data, time):
    y_pred = svc.predict(test_data)
    precision, recall, fscore, support = score(results, y_pred)

    print('precision: {}'.format(np.mean(precision)))
    print('recall: {}'.format(np.mean(recall)))
    print('fscore: {}'.format(np.mean(fscore)))
    print('support: {}'.format(support))
    data.append([index, label, dataset_label, time, np.mean(precision), np.mean(fscore), np.mean(recall)])


def save_file(name, text):
    with open(name, "w") as file:
        file.write(text)


if __name__ == '__main__':
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    df = df.drop('proto', axis=1)
    resultsColumns = ["index", "clf", "dataset", "time", "accuracy", "f1_score", "recall"]
    dataset_name = "PCA_5"
    data = []

    cat_columns = df.select_dtypes(['object']).columns
    df[cat_columns] = df[cat_columns].astype('category')
    print(cat_columns)
    for col in cat_columns:
        print(col)
        print(df[col].nunique())

    # only for ordinal values
    # df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    one_hot = pd.get_dummies(df[cat_columns], drop_first=False)
    df = df.drop(df[cat_columns].columns.values, axis=1)
    df = df.join(one_hot)

    print(df.label.value_counts())

    X = df.drop(['label'], axis=1).values
    y = df.label.values

    # X = preprocessing.StandardScaler().fit_transform(X)
    pca_components = PCA(n_components=5).fit_transform(X)
    X = pd.DataFrame(data=pca_components)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    print()
    print("Naive Bayes")
    clf = GaussianNB()
    scores = cross_val_score(clf, X, y, cv=3)
    print(np.mean(scores))

    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print_report(clf, X_test, "Naive Bayes", y_test)
    add_data(12, clf, X_test, "Naive Bayes", dataset_name, y_test, data, end-start)

    print()
    start = time.time()
    rfc = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0).fit(X_train, y_train)
    end = time.time()
    print_report(rfc, X_test, "Random Forest 100", y_test)
    add_data(13, rfc, X_test, "Random Forest 100", dataset_name, y_test, data, end - start)

    print()
    start = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=500).fit(X_train, y_train)
    end = time.time()
    print_report(mlp, X_test, "NN 5,5", y_test)
    add_data(14, mlp, X_test, "NN 5,5", dataset_name, y_test, data, end - start)

    print()
    start = time.time()
    svc_rbf = SVC(kernel='rbf', C=1, max_iter=2500).fit(X_train, y_train)
    end = time.time()
    print_report(svc_rbf, X_test, "SVM RBF c 1", y_test)
    add_data(15, svc_rbf, X_test, "SVM RBF c 1", dataset_name, y_test, data, end - start)

    results_df = pd.DataFrame(data, columns=resultsColumns)
    results_df.to_csv("dataframe_4.csv", index=False, header=True)

    '''
    print("SVM")
    svc_rbf = SVC(kernel='rbf', C=1)
    scores = cross_val_score(svc_rbf, X, y, cv=3)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    '''

    '''
    print("NN")
    mlp = MLPClassifier(hidden_layer_sizes=(10), max_iter=500)
    scores = cross_val_score(mlp, X, y, cv=3)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    '''

