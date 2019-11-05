from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def import_data(colnames):
    data = pd.read_csv('../cv6/iris.csv', sep=';', header=None, names=colnames)

    # Printing the dataset shape
    print("Dataset Length: ", len(data))
    print("Dataset Shape: ", data.shape)

    # Printing the dataset observations
    print("Dataset: ", data.head())
    return data


def plot_svm(models, x, y):
    h = .02  # step size in the mesh
    # create a mesh to plot in
    x_min, x_max = x.iloc[:, 0].min() - 1, x.iloc[:, 0].max() + 1
    y_min, y_max = x.iloc[:, 1].min() - 1, x.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['SVC with linear kernel C=1',
              'Linear SVC C = 10',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel']

    for i, clf in enumerate(models):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y, cmap=plt.cm.coolwarm)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    plt.show()


def print_report(svc, test_data, label, results):
    y_pred = svc.predict(test_data)
    print(label)
    print(classification_report(results, y_pred))


if __name__ == '__main__':
    colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
    dataset = import_data(colnames)
    # X = dataset.drop(['Class'], axis=1)
    X = dataset.iloc[:, :2]
    Y = dataset['Class']
    C = 1.0
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    svc_linear = SVC(kernel='linear', C=C).fit(X_train, y_train)
    svc_linear2 = SVC(kernel='linear', C=10).fit(X_train, y_train)
    svc_poly = SVC(kernel='poly', degree=3).fit(X_train, y_train)
    svc_rbf = SVC(kernel='rbf', C=10).fit(X_train, y_train)

    print_report(svc_linear, X_test, f"Linear C = {C}", y_test)
    print_report(svc_rbf, X_test, f"RBF C = {C}", y_test)

    plot_svm([svc_linear, svc_linear2, svc_rbf, svc_poly], X, Y)
