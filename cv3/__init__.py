import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster
import sklearn.metrics
import scipy.spatial
import pandas as pd


def plot_closest_distance(matrix):
    distance_matrix = scipy.spatial.distance_matrix(matrix, matrix)
    closest_neighbor_distance = np.where(distance_matrix == 0, 99, distance_matrix).min(axis=0)
    pd.Series(closest_neighbor_distance).hist(bins=40)
    plt.show()


def plot_db_scan(matrix, eps: float, min_samples: int):
    clusters_ = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(matrix)
    plt.scatter(matrix[:, 0], matrix[:, 1], c=clusters_.labels_)
    plt.show()
    return clusters_


def plot_kmeans(matrix, no_clusters: int):
    clusters_ = sklearn.cluster.KMeans(n_clusters=no_clusters).fit(matrix)
    plt.scatter(matrix[:, 0], matrix[:, 1], c=clusters_.labels_)
    plt.show()
    return clusters_


def print_clusters_size(clusters_):
    db_labels = clusters_.labels_
    labels, counts = np.unique(db_labels[db_labels >= 0], return_counts=True)
    noisy_labels, noisy_counts = np.unique(db_labels[db_labels < 0], return_counts=True)

    for i in range(len(labels)):
        print("Cluster: {0} size: {1} ".format(labels[i], counts[i]))
    if noisy_labels.size > 0:
        print("Noisy cluster size: {0} ".format(noisy_counts[0]))
    else:
        print("Noisy cluster size: 0")


if __name__ == '__main__':
    data_densegrid = np.loadtxt("densegrid.csv", delimiter=";")
    data_boxes = np.loadtxt("boxes.csv", delimiter=";")
    data_annulus = np.loadtxt("annulus.csv", delimiter=";")
    plot_closest_distance(data_boxes)
    plot_closest_distance(data_densegrid)

    '''
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
    '''
    # Agglomerative
    '''
    clusters = sklearn.cluster.AgglomerativeClustering(n_clusters=10, affinity="euclidean", linkage="single").fit(data)
    plt.scatter(data[:, 0], data[:, 1], c=clusters.labels_)
    plt.show()
    '''
    # K-means
    plot_kmeans(data_densegrid, 10)

    # DBSCAN Dense grid
    clusters = plot_db_scan(data_densegrid, 0.20, 10)
    # DBSCAN boxes
    clusters = plot_db_scan(data_boxes, 2.5, 2)
    print_clusters_size(clusters)

    # DBSCAN annulus
    clusters = plot_db_scan(data_annulus, 2, 2)



    print()
