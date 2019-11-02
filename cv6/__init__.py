import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def import_data():
    data = pd.read_csv('iris.csv', sep=';', header=None)

    # Printing the dataset shape
    print("Dataset Length: ", len(data))
    print("Dataset Shape: ", data.shape)

    # Printing the dataset observations
    print("Dataset: ", data.head())
    return data


def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum(
        [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy


def InfoGain(data, split_attribute_name, target_name="class"):
    total_entropy = entropy(data[target_name])

    vals, counts = np.unique(data[split_attribute_name], return_counts=True)

    Weighted_Entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
         for i in range(len(vals))])

    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain


def ID3(data, originaldata, features, target_attribute_name="class", parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[
            np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]

    elif len(features) == 0:
        return parent_node_class


    else:
        parent_node_class = np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

        item_values = [InfoGain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        tree = {best_feature: {}}

        features = [i for i in features if i != best_feature]


        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()

            subtree = ID3(sub_data, dataset, features, target_attribute_name, parent_node_class)

            tree[best_feature][value] = subtree

        return tree


def predict(query, tree, default=1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default

            result = tree[key][query[key]]
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result


def test(data, tree):
    # dictionary of features
    queries = data.iloc[:, :-1].to_dict(orient="records")
    predicted = pd.DataFrame(columns=["class"])

    # Calculate the prediction accuracy
    for i in range(len(data)):
        original_index = data.index[i]
        predicted.loc[original_index, "class"] = predict(queries[i], tree, 1)
    print('The prediction accuracy is: ', (np.sum(predicted["class"] == data["class"]) / len(data)) * 100, '%')


if __name__ == '__main__':
    dataset = pd.read_csv('zoo.data', sep=',', header=None,
                          names=['animal_name', 'hair', 'feathers', 'eggs', 'milk',
                                 'airbone', 'aquatic', 'predator', 'toothed', 'backbone',
                                 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize', 'class', ])
    dataset = dataset.drop('animal_name', axis=1)
    training_data, test_data = train_test_split(dataset, test_size=0.2)
    tree = ID3(training_data, training_data, training_data.columns[:-1])
    print(tree)
    test(test_data, tree)
