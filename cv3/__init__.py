import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster
import sklearn.metrics
import scipy.spatial
import pandas as pd

if __name__ == '__main__':
    data = np.loadtxt("densegrid.csv", delimiter=";")
    distance_matrix = scipy.spatial.distance_matrix(data, data)
    closest_neighbor_distance = np.where(distance_matrix == 0, 99, distance_matrix).min(axis=0)
    pd.Series(closest_neighbor_distance).hist(bins=40)
    plt.show()


    clustering_scores = []
    for k in range(2, 15):
        clusters = sklearn.cluster.KMeans(n_clusters=k).fit(data)
        clustering_scores.append({'k': k,
                                  'sse': clusters.inertia_,
                                  'silhouette': sklearn.metrics.cluster.silhouette_score(data, clusters.labels_)})

    df_score = pd.DataFrame.from_dict(clustering_scores, orient="columns")
    df_score = df_score.set_index('k')
    print(df_score)
    df_score.sse.plot()
    plt.show()
    df_score.silhouette.plot()
    plt.show()

    # Agglomerative
    clusters = sklearn.cluster.AgglomerativeClustering(n_clusters=10, affinity="euclidean", linkage="single").fit(data)
    plt.scatter(data[:, 0], data[:, 1], c=clusters.labels_)
    plt.show()
    # K-means
    clusters = sklearn.cluster.KMeans(n_clusters=10).fit(data)
    plt.scatter(data[:, 0], data[:, 1], c=clusters.labels_)
    plt.show()
    # DBSCAN
    clusters = sklearn.cluster.DBSCAN(eps=0.15, min_samples=10).fit(data)
    plt.scatter(data[:, 0], data[:, 1], c=clusters.labels_)
    plt.show()

    print()
