import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == '__main__':
    df1 = pd.read_csv('dataframe_1.csv', index_col=0)
    df2 = pd.read_csv('dataframe_2.csv', index_col=0)
    df3 = pd.read_csv('dataframe_3.csv', index_col=0)
    df4 = pd.read_csv('dataframe_4.csv', index_col=0)
    df = pd.concat([df1, df2, df3, df4]).reset_index()

    plt.title('Accuracy')
    sns.heatmap(df.pivot(index='dataset', columns='clf', values='accuracy'), annot=True, cmap="RdYlGn")
    plt.show()

    plt.title('F1 score')
    sns.heatmap(df.pivot(index='dataset', columns='clf', values='f1_score'), annot=True, cmap="RdYlGn")
    plt.show()

    plt.title('Recall')
    sns.heatmap(df.pivot(index='dataset', columns='clf', values='recall'), annot=True, cmap="RdYlGn")
    plt.show()

    plt.title('Fit time')
    sns.heatmap(df.pivot(index='dataset', columns='clf', values='time'), annot=True, cmap="RdYlGn_r")
    plt.show()

