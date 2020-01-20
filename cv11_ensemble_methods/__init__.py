from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':
    X = pd.read_csv('real_data_classification_X.csv', index_col=0)
    y = pd.read_csv('real_data_classification_y.csv', index_col=0).values.ravel()

    bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
    without_bagging = KNeighborsClassifier()

    rfc = RandomForestClassifier(n_estimators=100)
    tree = tree.DecisionTreeClassifier()
    MSE_rfc = cross_val_score(rfc, X, y, scoring="accuracy", cv=3)
    MSE_tree = cross_val_score(tree, X, y, scoring="accuracy", cv=3)
    mean_mse = np.mean(MSE_rfc)
    mean_mse_tree = np.mean(MSE_tree)
    print(f"RFC Accuracy: {mean_mse}")
    print(f"Tree Accuracy: {mean_mse_tree}")

    MSE_bagg = cross_val_score(bagging, X, y, scoring="accuracy", cv=3)
    MSE_bagg_without = cross_val_score(without_bagging, X, y, scoring="accuracy", cv=3)
    mean_bagg = np.mean(MSE_bagg)
    mean_bagg_without = np.mean(MSE_bagg_without)
    print(f"Bagging Accuracy: {mean_bagg}")
    print(f"NO Bagging Accuracy: {mean_bagg_without}")

    clf = AdaBoostClassifier(n_estimators=100)
    score_ada = cross_val_score(clf, X, y, scoring='accuracy', cv=3)
    mean_score = np.mean(score_ada)
    print(f"ADA BOOST Accuracy: {mean_score}")

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0)
    score_gbc = cross_val_score(clf, X, y, scoring="accuracy", cv=3)
    mean_score = np.mean(score_gbc)
    print(f"Gradient boosting Accuracy: {mean_score}")

