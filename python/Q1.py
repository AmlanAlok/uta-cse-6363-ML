import numpy as np
import math


def clean_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')


def fetch_data(file_name):

    with open(file_name, 'r') as f:
        input_data = f.readlines()
        clean_input = list(map(clean_data, input_data))
        f.close()

    return clean_input


def get_input_data(file_path):

    training_filename = file_path
    input_data_from_file = fetch_data(training_filename)
    td = np.array(input_data_from_file)
    training_input_data = add_numeric_labels(td)
    return training_input_data


def add_numeric_labels(td):
    y_data_points = td[:, 3]
    y_data = y_data_points.reshape(y_data_points.shape[0], 1)
    value_01 = np.unique(y_data, return_inverse=True)[1]        # (120,)
    p = value_01.reshape(value_01.shape[0], 1)
    x = np.concatenate((td[:, :3], p), axis=1)
    x = x.astype('float64')
    return x


def split_data(td):

    x = td[:, :3]
    y_label = td[:, 3]
    pass

    return x, y_label


def get_dist_mat(a):

    b = a[:, None]          # doing this for broadcasting
    c = a - b
    dist_mat = np.linalg.norm(c, axis=-1)

    '''
    Replacing 0 with the inf so that when finding min value, we do not get 0.
    We get the min distance amongst two separate points
    '''
    dist_mat[dist_mat == 0] = float('inf')
    return dist_mat


def find_index_of_next_min(dist_mat):

    min_num_index = np.unravel_index(dist_mat.argmin(), dist_mat.shape)
    row_idx, col_idx = min_num_index[0], min_num_index[1]

    print('Min value found = ', dist_mat[row_idx, col_idx])

    '''We do this so that next time the same min number is not returned'''
    dist_mat[row_idx, col_idx] = float('inf')
    dist_mat[col_idx, row_idx] = float('inf')

    return dist_mat, row_idx, col_idx


def get_dict(input_data):

    dict = {}
    num_of_data_points = input_data.shape[0]

    for i in range(num_of_data_points):

        d = {
            'input_array': input_data[i],
            'cluster_assigned': False,
            'cluster': -1
        }

        dict[i] = d

    return dict


def assign_cluster(index, cluster_index, dict):
    dict[index]['cluster_assigned'] = True
    dict[index]['cluster'] = cluster_index


def only_1_point_in_cluster(x1_idx, x2_idx, d, cluster_dict, cluster_count):
    cluster_index = d[x1_idx]['cluster']
    cluster_dict[cluster_index] = cluster_dict[cluster_index] + cluster_dict[x2_idx]
    cluster_dict[x2_idx] = []
    cluster_count -= 1
    assign_cluster(x2_idx, cluster_index, d)
    return cluster_count

def add_indexes_to_same_cluster(x1_idx, x2_idx, d, cluster_dict, cluster_count):

    cluster_assigned = 'cluster_assigned'

    x1_assigned = d[x1_idx][cluster_assigned]
    x2_assigned = d[x2_idx][cluster_assigned]
    print('add')
    """ When both points are not assigned to any cluster"""
    if not x1_assigned and not x2_assigned:

        # cluster_dict[x1_idx].append(x2_idx)
        cluster_dict[x1_idx] = cluster_dict[x1_idx] + cluster_dict[x2_idx]
        cluster_dict[x2_idx] = []
        cluster_count -= 1

        assign_cluster(x1_idx, x1_idx, d)
        assign_cluster(x2_idx, x1_idx, d)

    # ''' When both points are assigned to any cluster'''
    elif x1_assigned and x2_assigned:

        cluster_index1 = d[x1_idx]['cluster']
        cluster_index2 = d[x2_idx]['cluster']

        if cluster_index1 != cluster_index2:
            cluster_dict[cluster_index1] = cluster_dict[cluster_index1] + cluster_dict[cluster_index2]

            for i in cluster_dict[cluster_index2]:
                assign_cluster(i, cluster_index1, d)

            cluster_dict[cluster_index2] = []
            cluster_count -= 1
        else:
            """Edge case: Shortest distance is between points already in the same cluster"""
            print('x1 and x2 are already part of the same cluster')

    # """1st point is in cluster and point 2 is not"""
    elif x1_assigned and not x2_assigned:

        cluster_count = only_1_point_in_cluster(x1_idx, x2_idx, d, cluster_dict, cluster_count)
    # """1st point is not in cluster and point 2 is in cluster"""
    elif not x1_assigned and x2_assigned:

        cluster_count = only_1_point_in_cluster(x2_idx, x1_idx, d, cluster_dict, cluster_count)
        pass
    # """ When only one point is assigned to any cluster"""

    return d, cluster_count


def create_cluster_foreach_data_point(input_data):

    cluster_dict = {}
    num_of_data_points = input_data.shape[0]

    for i in range(num_of_data_points):
        cluster_dict[i] = [i]

    return cluster_dict


def main():

    file_path = '../data/120_data_points.txt'
    training_input_data = get_input_data(file_path=file_path)

    """Splitting input data into (X, Y)"""
    input_data, y_label = split_data(training_input_data)

    dict = get_dict(input_data)
    cluster_dict = create_cluster_foreach_data_point(input_data)

    '''Calculating Distance Matrix'''
    dist_mat = get_dist_mat(input_data)

    cluster_count = len(cluster_dict)
    interested_clusters = [2, 4, 6, 8]
    interested_dict = {}

    while cluster_count > 1:
        dist_mat, x_idx, y_idx = find_index_of_next_min(dist_mat)
        dict, cluster_count = add_indexes_to_same_cluster(x_idx, y_idx, dict, cluster_dict, cluster_count)
        if cluster_count in interested_clusters:
            interested_dict[cluster_count] = cluster_dict.copy()




    pass


if __name__ == "__main__":
    main()