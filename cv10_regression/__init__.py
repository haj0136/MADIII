import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':
    df = pd.read_csv('bike.csv', parse_dates=['dteday'], index_col=0)
    df['dteday'] = df['dteday'].dt.dayofyear

    X = df.drop(['cnt'], axis=1).values
    y = df.cnt.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    reg01 = linear_model.Ridge(alpha=.01).fit(X_train, y_train)
    reg10 = linear_model.Ridge(alpha=100).fit(X_train, y_train)
    reg = linear_model.Ridge(alpha=0.001)
    MSE = cross_val_score(reg, X, y, scoring="neg_mean_squared_error", cv=3)
    mean_mse = np.mean(MSE)
    print(f"RIDGE MSE: {mean_mse}")
    lasso01 = linear_model.Lasso(alpha=.01).fit(X_train, y_train)
    lasso100 = linear_model.Lasso(alpha=100).fit(X_train, y_train)



    Ridge_test_score = reg01.score(X_test, y_test)
    Ridge_test_score100 = reg10.score(X_test, y_test)

    lasso_test_score = lasso01.score(X_test, y_test)
    lasso_test_score100 = lasso100.score(X_test, y_test)

    print("ridge regression test score low alpha:", Ridge_test_score)
    print("ridge regression test score high alpha:", Ridge_test_score100)

    print("lasso regression test score low alpha:", lasso_test_score)
    print("lasso regression test score high alpha:", lasso_test_score100)


