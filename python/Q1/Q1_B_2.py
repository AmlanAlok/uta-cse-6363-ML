import numpy as np
import math


class TreeNode:

    def __init__(self, feature_name, feature_index, threshold):
        self.feature_name = feature_name
        self.feature_index = feature_index
        self.threshold = threshold
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


def cal_gain_ration_for_data_set(sorted_input, root_entropy, i, j):

    '''creating test threshold'''
    c = (sorted_input[j, i] + sorted_input[j - 1, i]) / 2

    '''splitting data according to this threshold'''
    count = np.count_nonzero(sorted_input[:, i] < c)

    '''splitting the label column'''
    p1, p2 = sorted_input[:count, 3], sorted_input[count:, 3]

    '''Counting M and W in each split'''
    left_m, left_w = len(p1[p1 == 0]), len(p1[p1 == 1])
    right_m, right_w = len(p2[p2 == 0]), len(p2[p2 == 1])

    '''Calculating Information Gain'''
    information_gain = gain(root_entropy, [left_m, left_w], [right_m, right_w])

    '''Calculating Gain Ratio'''
    gain_ratio = cal_gain_ratio(information_gain, [len(p1), len(p2)])

    # '''Adding Gain Ratio to a list'''
    # gain_ratio_list.append(gain_ratio)

    '''Storing the Gain Ratio, threshold, element count on splits'''
    # gain_ratio_array[j - 1] = [gain_ratio, c, len(p1), len(p2)]
    return [gain_ratio, c, len(p1), len(p2)]


def main():
    # filename = 'datasets/Q1_b_training_data.txt'
    filename = '../../data/Q1_C_training.txt'
    height_idx, weight_idx, age_idx, label_idx = 0, 1, 2, 3
    feature_indexes = [0, 1, 2]

    feature_dict = {
        0: 'height',
        1: 'weight',
        2: 'age'
    }

    rootNode: TreeNode

    input_data = fetch_data(filename)

    td = np.array(input_data)
    input = add_numeric_labels(td)
    best_gain_ratio_across_features = []
    best_threshold_across_features = []

    depth = 8

    for d in range(depth):

        for i in feature_indexes:
            '''sorting the input by feature value'''
            sorted_input = input[input[:, i].argsort()]

            '''getting the sorted label column'''
            label_col = sorted_input[:, 3]

            '''counting number of M and W'''
            m_count, w_count = len(label_col[label_col == 0]), len(label_col[label_col == 1])

            '''Calculating root entropy'''
            root_entropy = cal_entropy([m_count, w_count])

            '''Calculating the size of training data'''
            training_data_size = sorted_input.shape[0]

            gain_ratio_list = []
            # gain_ratio_array = np.array([])
            gain_ratio_array = np.empty((training_data_size-1, 4))

            for j in range(1, training_data_size):

                gain_ratio_array[j - 1] = cal_gain_ration_for_data_set(sorted_input, root_entropy, i, j)

                pass

            # max_information_gain = gain_ratio_list.index(max(gain_ratio_list))
            '''Finding the index of the max Gain Ratio'''
            max_index = gain_ratio_array[:, 0].argmax()

            '''Getting the threshold which gave the Max Gain Ratio'''
            best_threshold_feature = gain_ratio_array[max_index, 1]

            '''Storing Best Gain Ratio for this feature'''
            best_gain_ratio_across_features.append(gain_ratio_array[max_index, 0])

            '''Storing Best Threshold for this feature'''
            best_threshold_across_features.append(best_threshold_feature)

            pass

        best_gain_ratio_index = best_gain_ratio_across_features.index(max(best_gain_ratio_across_features))
        best_gain_ratio = best_gain_ratio_across_features[best_gain_ratio_index]
        best_threshold = best_threshold_across_features[best_gain_ratio_index]

        if d == 1:
            rootNode = TreeNode(feature_name=feature_dict[best_gain_ratio_index],
                          feature_index=best_gain_ratio_index,
                          threshold=best_threshold)
        else:
            rootNode.add_child()



        # height_data, weight_data, age_data, y_data = separate_input_output(input_data)
        # y_data_01 = change_y_data(y_data)

        pass


if __name__ == "__main__":
    main()