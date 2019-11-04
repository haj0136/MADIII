from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def import_data(colnames):
    data = pd.read_csv('../cv6/iris.csv', sep=';', header=None, names=colnames)

    # Printing the dataset shape
    print("Dataset Length: ", len(data))
    print("Dataset Shape: ", data.shape)

    # Printing the dataset observations
    print("Dataset: ", data.head())
    return data


if __name__ == '__main__':
    colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
    dataset = import_data(colnames)
    X = dataset.drop(['Class'], axis=1)
    Y = dataset['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    svc_poly = SVC(kernel='poly', degree=8)
    svc_poly.fit(X_train, y_train)

    y_pred = svc_poly.predict(X_test)
    print("Polynomial")
    print()
    print(classification_report(y_test, y_pred))

    svc_rbf = SVC(kernel='rbf', C=10)
    svc_rbf.fit(X_train, y_train)
    y_pred = svc_rbf.predict(X_test)
    print("RBF")
    print()
    print(classification_report(y_test, y_pred))
