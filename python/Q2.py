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
    y = np.array([y_label])
    # y = y.reshape((y.shape[1], y.shape[0]))
    y = y.transpose()
    return x, y


def partition_input_data(x, break_point):
    supervised_data = x[:break_point, :]
    unsupervised_data = x[break_point:, :]
    return supervised_data, unsupervised_data


def partition_label_data(x, break_point):
    supervised_data = x[:break_point]
    unsupervised_data = x[break_point:]
    return supervised_data, unsupervised_data


def get_distance(input_data, test_input):

    a = [None]*len(input_data)
    i = 0

    for dp in input_data:
        cartesian_distance = math.sqrt(
            pow(dp['input']['height']-test_input['input']['height'], 2) +
            pow(dp['input']['weight']-test_input['input']['weight'],2) +
            pow(dp['input']['age']-test_input['input']['age'], 2))

        a[i] = {
            'index': dp['index'],
            'cartesian_distance': cartesian_distance,
            'output': dp['output']
        }

        i += 1

    sorted_a = sorted(a, key=lambda d: d['cartesian_distance'])

    # print(sorted_a)

    return sorted_a


def calc_distance_from_given_point(x_supervised, y_supervised, dp):

    xy_supervised = np.concatenate((x_supervised, y_supervised), axis=1)
    supervised_size = xy_supervised.shape[0]
    dist_mat = np.full((supervised_size, 3), float('inf'))
    dist_index = 1
    inverse_dist_index = 2

    for i in range(supervised_size):
        dist_mat[i][0] = i
        d = np.linalg.norm(x_supervised[i] - dp)
        dist_mat[i][dist_index] = d
        dist_mat[i][inverse_dist_index] = 1/d

    xy_supervised_dist = np.concatenate((dist_mat, xy_supervised), axis=1)
    xy_supervised_dist = xy_supervised_dist[xy_supervised_dist[:, dist_index].argsort()]
    pass
    return xy_supervised_dist


def only_supervised_approach(x_supervised, y_supervised, x_unsupervised, y_unsupervised):

    for dp in x_unsupervised:
        # print(dp)
        xy_supervised_dist = calc_distance_from_given_point(x_supervised, y_supervised, dp)

    pass


def main():

    partition = 20
    file_path = '../data/120_data_points.txt'
    training_input_data = get_input_data(file_path=file_path)

    """Splitting input data into (X, Y)"""
    input_data, y_label = split_data(training_input_data)

    '''Splitting the input data into supervised and unsupervised'''
    x_supervised, x_unsupervised = partition_input_data(input_data, partition)
    y_supervised, y_unsupervised = partition_label_data(y_label, partition)

    # y_supervised.reshape((20,1))
    only_supervised_approach(x_supervised, y_supervised, x_unsupervised, y_unsupervised)
    pass


if __name__ == "__main__":
    main()