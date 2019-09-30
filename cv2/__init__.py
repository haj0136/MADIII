import pandas as pd
from matplotlib import pyplot as plt
from cv2.Agglomerative_clustering import Agglomerative


if __name__ == '__main__':
    df = pd.read_csv('clusters3.csv', sep=';', header=None)
    df.columns = ['x', 'y']

    clustering_algorithm = Agglomerative(n_clusters=5, linkage="single")
    index_cluster_list = clustering_algorithm.get_clusters(df)
    df['cluster'] = index_cluster_list

    print('Results size', df.shape)
    df.plot.scatter(x='x', y='y', c='cluster', colormap='Set1')
    plt.show()
