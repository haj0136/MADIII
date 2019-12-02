import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df = pd.read_csv('bike.csv', parse_dates=['dteday'], index_col=0)
    df['dteday'] = df['dteday'].dt.dayofyear

    X = df.drop(['cnt'], axis=1).values
    y = df.cnt.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    reg = linear_model.Ridge(alpha=.05).fit(X_train, y_train)
    reg10 = linear_model.Ridge(alpha=10).fit(X_train, y_train)

    Ridge_train_score = reg.score(X_train, y_train)
    Ridge_test_score = reg.score(X_test, y_test)
    Ridge_train_score100 = reg10.score(X_train, y_train)
    Ridge_test_score100 = reg10.score(X_test, y_test)

    print("ridge regression train score low alpha:", Ridge_train_score)
    print("ridge regression test score low alpha:", Ridge_test_score)
    print("ridge regression train score high alpha:", Ridge_train_score100)
    print("ridge regression test score high alpha:", Ridge_test_score100)
