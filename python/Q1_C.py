import numpy as np


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

    pass


if __name__ == "__main__":
    main()