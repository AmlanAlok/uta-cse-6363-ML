import numpy as np
import math
# class Record


class TreeNode:
    def __init__(self, data):
        self.data = data
        self.children = []
        self.parent = None

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def print_tree(self):
        gap = ' ' * 2 * self.get_level() + '|--'
        print(gap + self.data)
        if self.children:
            for child in self.children:
                child.print_tree()


def clean_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')


def fetch_data(file_name):
    # print('inside func '+ inspect.stack()[0][3])

    with open(file_name, 'r') as f:
        input_data = f.readlines()
        # print(type(input_data ))
        # print('Number of data points =', len(input_data ))

        clean_input = list(map(clean_data, input_data))

        f.close()

    return clean_input


def change_y_data(y_data):
    value_01 = np.unique(y_data, return_inverse=True)[1]
    return value_01

'''Uses all Data'''


def separate_input_output(input_data):

    td = np.array(input_data)

    height_data_points = td[:, 0]
    height_data = height_data_points.reshape(height_data_points.shape[0], 1)
    height_data = height_data.astype('float64')

    weight_data_points = td[:, 1]
    weight_data = weight_data_points.reshape(weight_data_points.shape[0], 1)
    weight_data = weight_data.astype('float64')

    age_data_points = td[:, 2]
    age_data = age_data_points.reshape(age_data_points.shape[0], 1)
    age_data = age_data.astype('int64')

    y_data_points = td[:, 3]


    return height_data, weight_data, age_data, y_data

def add_numeric_labels(td):
    y_data_points = td[:, 3]
    y_data = y_data_points.reshape(y_data_points.shape[0], 1)
    value_01 = np.unique(y_data, return_inverse=True)[1]        # (120,)
    p = value_01.reshape(value_01.shape[0], 1)
    x = np.concatenate((td[:, :3], p), axis=1)
    x = x.astype('float64')
    return x


def cal_entropy(x):

    s = sum(x)
    e = 0

    for i in x:

        if i > 0:
            e += (i / s) * math.log(i / s, 2)

    e = e*-1
    return e


def gain(root_entropy, left_split_array, right_split_array):

    left_sum = sum(left_split_array)
    left_entropy = cal_entropy(left_split_array)

    right_sum = sum(right_split_array)
    right_entropy = cal_entropy(right_split_array)

    total_sum = left_sum + right_sum

    child_entropy = (left_sum/total_sum)*left_entropy + (right_sum/total_sum)*right_entropy

    information_gain = root_entropy - child_entropy

    return information_gain


def cal_gain_ratio(gain, split_array):

    s = sum(split_array)
    d = 0

    for i in split_array:
        if i > 0:
            d += (i/s) * math.log(i/s, 2)

    gain_ratio = gain/(-1 * d)

    return gain_ratio


def main():
    # filename = 'datasets/Q1_b_training_data.txt'
    filename = '../../data/Q1_C_training.txt'
    height_idx, weight_idx, age_idx, label_idx = 0, 1, 2, 3
    feature_indexes = [0, 1, 2]

    input_data = fetch_data(filename)

    td = np.array(input_data)
    input = add_numeric_labels(td)
    best_gain_ratio_across_features = []

    for i in feature_indexes:
        sorted_input = input[input[:, i].argsort()]

        label_col = sorted_input[:, 3]
        m_count, w_count = len(label_col[label_col == 0]), len(label_col[label_col == 1])

        root_entropy = cal_entropy([m_count, w_count])

        training_data_size = sorted_input.shape[0]

        gain_ratio_list = []
        # gain_ratio_array = np.array([])
        gain_ratio_array = np.empty((training_data_size-1, 4))

        for j in range(1, training_data_size):

            left_m, left_w = 0, 0
            right_m, right_w = 0, 0

            c = (sorted_input[j, i] + sorted_input[j-1, i])/2

            count = np.count_nonzero(sorted_input[:, i] < c )

            # p = np.vsplit(sorted_input, 2)
            p1, p2 = sorted_input[:count, 3], sorted_input[count:, 3]

            left_m, left_w = len(p1[p1 == 0]), len(p1[p1 == 1])
            right_m, right_w = len(p2[p2 == 0]), len(p2[p2 == 1])

            information_gain = gain(root_entropy, [left_m, left_w], [right_m, right_w])

            gain_ratio = cal_gain_ratio(information_gain, [len(p1), len(p2)])

            gain_ratio_list.append(gain_ratio)
            gain_ratio_array[j-1] = [gain_ratio, c, len(p1), len(p2)]

            pass

        max_information_gain = gain_ratio_list.index(max(gain_ratio_list))
        max_index = gain_ratio_array[:, 0].argmax()
        best_threshold = gain_ratio_array[max_index, 1]
        best_gain_ratio_across_features.append(gain_ratio_array[max_index, 0])

        pass

    # height_data, weight_data, age_data, y_data = separate_input_output(input_data)
    # y_data_01 = change_y_data(y_data)

    pass


if __name__ == "__main__":
    main()