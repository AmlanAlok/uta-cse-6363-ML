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

    b = a[:, None]
    c = a - b
    d = np.linalg.norm(c, axis=-1)

    return d

def main():

    training_filename = '../data/120_data_points.txt'

    input_data_from_file = fetch_data(training_filename)
    td = np.array(input_data_from_file)
    training_input_data = add_numeric_labels(td)

    inp, y_label = split_data(training_input_data)

    dist_mat = get_dist_mat(inp)

    pass


if __name__ == "__main__":
    main()