import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == '__main__':
    df1 = pd.read_csv('SAHeart_RF100.csv', index_col=0)
    df2 = pd.read_csv('SAHeart_MYRF100.csv', index_col=0)
    # df = pd.concat([df1, df2, df3, df4]).reset_index()

    plt.title('SKL Accuracy')
    sns.heatmap(df1.pivot(index='max_depth', columns='min_samples', values='accuracy'), annot=True, cmap="RdYlGn")
    plt.show()

    plt.title('SKL F1 score')
    sns.heatmap(df1.pivot(index='max_depth', columns='min_samples', values='f1_score'), annot=True, cmap="RdYlGn")
    plt.show()

    plt.title('SKL Recall')
    sns.heatmap(df1.pivot(index='max_depth', columns='min_samples', values='recall'), annot=True, cmap="RdYlGn")
    plt.show()

    plt.title('SKL Fit time')
    sns.heatmap(df1.pivot(index='max_depth', columns='min_samples', values='time'), annot=True, cmap="RdYlGn_r")
    plt.show()

    # My Random forest
    plt.title('My implementation Accuracy')
    sns.heatmap(df2.pivot(index='max_depth', columns='min_samples', values='accuracy'), annot=True, cmap="RdYlGn")
    plt.show()

    plt.title('My implementation F1 score')
    sns.heatmap(df2.pivot(index='max_depth', columns='min_samples', values='f1_score'), annot=True, cmap="RdYlGn")
    plt.show()

    plt.title('My implementation Recall')
    sns.heatmap(df2.pivot(index='max_depth', columns='min_samples', values='recall'), annot=True, cmap="RdYlGn")
    plt.show()

    plt.title('My implementation Fit time')
    sns.heatmap(df2.pivot(index='max_depth', columns='min_samples', values='time'), annot=True, cmap="RdYlGn_r")
    plt.show()

