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


def print_report(svc, test_data, label, results):
    y_pred = svc.predict(test_data)
    print(label)
    print(classification_report(results, y_pred))


if __name__ == '__main__':
    df = pd.read_csv('UNSW_NB15_training-set.csv', index_col=0)
    df = df.drop('proto', axis=1)

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

    X = preprocessing.StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    print()
    print("Naive Bayes")
    clf = GaussianNB()
    scores = cross_val_score(clf, X, y, cv=3)
    print(np.mean(scores))

    clf.fit(X_train, y_train)
    print_report(clf, X_test, "Naive Bayes", y_test)

    print()
    rfc = RandomForestClassifier(n_estimators=1000, max_depth=2, random_state=0).fit(X_train, y_train)
    print_report(rfc, X_test, "Random Forest 1000", y_test)

    print()
    mlp = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=500).fit(X_train, y_train)
    print_report(mlp, X_test, "NN 5,5", y_test)

    print()
    svc_rbf = SVC(kernel='rbf', C=10).fit(X_train, y_train)
    print_report(svc_rbf, X_test, "SVM RBF c 10", y_test)
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

