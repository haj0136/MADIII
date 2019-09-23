import pandas as pd
from matplotlib import pyplot as plt



if __name__ == '__main__':
    df = pd.read_csv('clusters3.csv', sep=';', header=None)
    df.columns = ['x', 'y']
    clusters = range(1, len(df.index) + 1)
    df['cluster'] = clusters

    print('Results size', df.shape)
    df.plot.scatter(x='x', y='y', c='cluster', colormap='Set1')
    plt.show()
