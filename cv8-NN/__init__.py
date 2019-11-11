import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


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
    dataset = import_data(colnames, '../cv6/nonsep.csv')
    X = dataset.drop(['class'], axis=1)
    Y = dataset['class']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100), max_iter=1000)
    mlp.fit(X_train, y_train.values.ravel())

    predictions = mlp.predict(X_test)
    print(classification_report(y_test, predictions))
