import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score


def import_data(colnames, path):
    data = pd.read_csv(path, sep=';', header=None, names=colnames)

    # Printing the dataset shape
    print("Dataset Length: ", len(data))
    print("Dataset Shape: ", data.shape)

    # Printing the dataset observations
    print("Dataset: ", data.head())
    return data


if __name__ == '__main__':
    colnames = ['x1', 'x2', 'class']
    iris_colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = import_data(colnames, '../cv6/nonsep.csv')
    iris_dataset = import_data(iris_colnames, '../cv6/iris.csv')
    X = iris_dataset.drop(['class'], axis=1)
    Y = iris_dataset['class']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1500)
    scores = cross_val_score(mlp, X, Y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    '''
    mlp.fit(X_train, y_train)

    predictions = mlp.predict(X_test)
    print(classification_report(y_test, predictions))
    '''
