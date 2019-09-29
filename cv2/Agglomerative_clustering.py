import math
import pandas as pd
import enum


class Linkage(enum.Enum):
    single = 1
    complete = 2


class Agglomerative:
    labels_ = []

    def __init__(self, n_clusters=2, linkage='single'):
        if linkage not in Linkage._member_names_:
            raise ValueError("Unknown linkage type: %s. Valid options are %s" % (linkage, str([e.name for e in Linkage])))
        self.linkage = Linkage[linkage]
        self.n_clusters = n_clusters
        self.distance_matrix = []

    def get_clusters(self, X):
        if self.n_clusters <= 0:
            raise ValueError("n_clusters should be an integer greater than 0.")

        # Initialize data to X
        self.data = X
        # Treat every instance as single cluster
        self.init_cluster()
        # Calculate distance matrix
        self.distance_matrix = []
        self.init_distance_matrix()

        while len(self.cluster) > self.n_clusters:
            # Find index of "minimum" value in distance matrix
            idxmin = self.distance_matrix_idxmin()
            # Cluster two instance based on "minimum" value
            self.update_cluster(idxmin[0], idxmin[1])
            # Update distance matrix
            self.distance_matrix_update(idxmin[0], idxmin[1])
            print(len(self.cluster))

        print("Clustering linkage: " + self.linkage.name)
        self.generate_label()
        return self.labels_

    def init_cluster(self):
        raw_data = self.data.values
        self.cluster = [[x] for x, val in enumerate(raw_data)]

    def init_distance_matrix(self):
        raw_data = self.data.values
        for index, eval_instance in enumerate(raw_data):
            distance_array = []
            for pair_instance in raw_data[index + 1:]:
                distance = self.calculate_distance(eval_instance, pair_instance)
                distance_array.append(distance)
            if distance_array:
                self.distance_matrix.append(distance_array)

    def calculate_distance(self, instance1, instance2):
        distance = 0
        for index, val in enumerate(instance1):
            attr_distance = (val - instance2[index]) ** 2
            distance += attr_distance

        return math.sqrt(distance)

    def distance_matrix_idxmin(self):
        min_val = self.distance_matrix[0][0]
        min_idx = [0, 0]
        for i, val_i in enumerate(self.distance_matrix):
            for j, val_j in enumerate(val_i):
                if min_val > val_j:
                    min_val = val_j
                    min_idx = [i, j]
        min_idx[1] = min_idx[0] + j + 1
        # print(min_idx)
        return min_idx

    def get_all_cluster_member(self, cluster):
        if len(cluster) == 1:
            return cluster
        else:
            member = []
            for subcluster in cluster:
                member = member+self.get_all_cluster_member(subcluster)
            return member

    def distance_matrix_update(self, instance1, instance2):
        self.distance_matrix.append([])
        coordinate_to_delete = []

        for index, val in enumerate(self.distance_matrix):
            if index != instance1 and index != instance2:
                coordinate = self.transform_matrix_coordinate(index, instance1)
                coordinate_compare = self.transform_matrix_coordinate(index, instance2)
                cell_x = coordinate[0]
                cell_y = coordinate[1]
                cell_x_compare = coordinate_compare[0]
                cell_y_compare = coordinate_compare[1]

                if self.linkage.name == "complete":
                    val_update = max(self.distance_matrix[cell_x][cell_y],
                                    self.distance_matrix[cell_x_compare][cell_y_compare])
                else:
                    val_update = min(self.distance_matrix[cell_x][cell_y],
                                    self.distance_matrix[cell_x_compare][cell_y_compare])
                self.distance_matrix[cell_x][cell_y] = val_update
                self.distance_matrix[cell_x_compare][cell_y_compare] = 0
                coordinate_to_delete.append(coordinate_compare)
        coord_to_del = self.transform_matrix_coordinate(instance1, instance2)
        coordinate_to_delete.append(coord_to_del)
        # Delete all 0-valued cells
        for index, val in enumerate(coordinate_to_delete):
            del self.distance_matrix[val[0]][val[1]]
            for j, next_vals in enumerate(coordinate_to_delete[index + 1:]):
                if next_vals[0] == val[0]:
                    coordinate_to_delete[index + 1 + j][1] -= 1
        # Delete all empty cluster
        cluster_length = len(self.distance_matrix)
        cell_row_idx = 0
        while cell_row_idx < cluster_length:
            if not self.distance_matrix[cell_row_idx]:
                del self.distance_matrix[cell_row_idx]
                cell_row_idx -= 1
                cluster_length -= 1
            cell_row_idx += 1

    def transform_matrix_coordinate(self, cell_x, cell_y):
        coordinate = [min(cell_x, cell_y), max(cell_x, cell_y)]
        coordinate[1] = coordinate[1] - (coordinate[0] + 1)
        # print(coordinate)
        return coordinate

    def update_cluster(self, index_instance1, index_instance2):
        self.cluster[index_instance1] = [self.cluster[index_instance1], self.cluster[index_instance2]]
        del self.cluster[index_instance2]

    def set_label(self, component_tree=None, label=None):
        if not isinstance(component_tree, list):
            self.labels_.insert(component_tree, label)
        else:
            for component in component_tree:
                self.set_label(component, label)

    def generate_label(self):
        for i, cluster in enumerate(self.cluster):
            self.set_label(cluster, i)
