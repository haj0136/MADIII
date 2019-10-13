import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster
import sklearn.metrics
import scipy.spatial
import pandas as pd


def pd_options():
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)



def plot_closest_distance(matrix):
    distance_matrix = scipy.spatial.distance_matrix(matrix, matrix)
    closest_neighbor_distance = np.where(distance_matrix == 0, 99, distance_matrix).min(axis=0)
    pd.Series(closest_neighbor_distance).hist(bins=40)
    plt.show()


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
    pd_options()
    df = pd.read_csv('happiness_report_2017.csv')

    df_numerical = df.drop(['Country', 'Happiness.Rank'], axis=1)
    df_features = df_numerical.drop(['Happiness.Score', 'Whisker.high', 'Whisker.low'], axis=1)

    clustering_scores = []
    for k in range(2, 15):
        clusters = sklearn.cluster.KMeans(n_clusters=k).fit(df_features)
        clustering_scores.append({'k': k,
                                  'sse': clusters.inertia_,
                                  'silhouette': sklearn.metrics.cluster.silhouette_score(df_features, clusters.labels_)})

    df_score = pd.DataFrame.from_dict(clustering_scores, orient="columns")
    df_score = df_score.set_index('k')
    print(df_score)
    df_score.sse.plot()
    plt.show()
    df_score.silhouette.plot()
    plt.show()

    clusters_dbscan = sklearn.cluster.DBSCAN(eps=0.34, min_samples=2).fit(df_features)
    clusters_kmeans = sklearn.cluster.KMeans(n_clusters=4).fit(df_features)
    clusters_agglomerative = sklearn.cluster.AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="complete").fit(df_features)
    clusters_agglomerative_ward = sklearn.cluster.AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="ward").fit(df_features)

    # df_labels = pd.DataFrame(clusters_kmeans.labels_)
    df_features.insert((df_features.shape[1]), "kmeans", clusters_kmeans.labels_)
    df_features.insert((df_features.shape[1]), "dbscan", clusters_dbscan.labels_)
    df_features.insert((df_features.shape[1]), "agglomerative", clusters_agglomerative.labels_)
    df_features.insert((df_features.shape[1]), "agglo.ward", clusters_agglomerative_ward.labels_)
    print(df_features.groupby(['kmeans']).mean())

    df.insert((df_features.shape[1]), "kmeans", clusters_kmeans.labels_)
    print(df[df.kmeans == 2])

    print_clusters_size(clusters_dbscan)

