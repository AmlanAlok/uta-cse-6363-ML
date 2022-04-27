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
    return x, y_label


def partition_input_data(x, break_point):

    supervised_data = x[:break_point, :]
    unsupervised_data = x[break_point:, :]

    return supervised_data, unsupervised_data

def partition_label_data(x, break_point):

    supervised_data = x[:break_point]
    unsupervised_data = x[break_point:]

    return supervised_data, unsupervised_data


def main():

    partition = 20
    file_path = '../data/120_data_points.txt'
    training_input_data = get_input_data(file_path=file_path)

    """Splitting input data into (X, Y)"""
    input_data, y_label = split_data(training_input_data)

    '''Splitting the input data into supervised and unsupervised'''
    x_supervised, x_unsupervised = partition_input_data(input_data, partition)
    y_supervised, y_unsupervised = partition_label_data(y_label, partition)

    pass


if __name__ == "__main__":
    main()