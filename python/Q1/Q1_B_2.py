import numpy as np
import math


class TreeNode:

    # stats = None
    # leftNode = None
    # rightNode = None
    # parent = None
    # dataset = None
    # leaf = None
    # decision = None
    # depth = None

    def __init__(self):
        self.stats = None
        self.leftNode = None
        self.rightNode = None
        self.parent = None
        self.dataset = None
        self.leaf = False
        self.decision = None
        self.depth = None
        # self.feature_name = feature_name
        # self.feature_index = feature_index
        # self.best_gain_ratio = best_gain_ratio
        # self.threshold = threshold
        # self.split_point = split_point
        # self.parent = None

    # def add_child(self, child):
    #     self.children.append(child)
    #     child.parent = self

    # def print_tree(self):
    #     gap = ' ' * 2 * self.get_level() + '|--'
    #     print(gap + self.data)
    #     if self.children:
    #         for child in self.children:
    #             child.print_tree()

    def add_stats(self, feature_name, feature_index, best_gain_ratio, feature_threshold, split_point):

        info_dict = {
            "feature_name": feature_name,
            "feature_index": feature_index,
            "best_gain_ratio": best_gain_ratio,
            "feature_threshold": feature_threshold,
            "split_point": split_point
        }

        self.stats = info_dict


def clean_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')


def fetch_data(file_name):

    with open(file_name, 'r') as f:
        input_data = f.readlines()
        clean_input = list(map(clean_data, input_data))
        f.close()

    return clean_input


def change_y_data(y_data):
    value_01 = np.unique(y_data, return_inverse=True)[1]
    return value_01


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

    try:
        gain_ratio = gain/(-1 * d)
        return gain_ratio
    except:
        print('Error: cal_gain_ratio')
        print('split_array =',split_array)
        print('sum =', s)
        print('d =', d)
        raise Exception('Error: cal_gain_ratio')


def cal_gain_ration_for_data_set(sorted_input, root_entropy, i, j):

    '''creating test threshold'''
    c = (sorted_input[j, i] + sorted_input[j - 1, i]) / 2

    '''splitting data according to this threshold'''
    count = np.count_nonzero(sorted_input[:, i] <= c)

    '''splitting the label column'''
    p1, p2 = sorted_input[:count, 3], sorted_input[count:, 3]

    '''Counting M and W in each split'''
    left_m, left_w = len(p1[p1 == 0]), len(p1[p1 == 1])
    right_m, right_w = len(p2[p2 == 0]), len(p2[p2 == 1])

    '''Calculating Information Gain'''
    information_gain = gain(root_entropy, [left_m, left_w], [right_m, right_w])

    '''Calculating Gain Ratio'''
    if len(p1) == 0 or len(p2) == 0:
        return None

    gain_ratio = cal_gain_ratio(information_gain, [len(p1), len(p2)])


    # '''Adding Gain Ratio to a list'''
    # gain_ratio_list.append(gain_ratio)

    '''Storing the Gain Ratio, threshold, element count on splits'''
    # gain_ratio_array[j - 1] = [gain_ratio, c, len(p1), len(p2)]
    return [gain_ratio, c, count, len(p1), len(p2)]


def choose_feature_and_threshold(input_data, i):

    '''sorting the input by feature value'''
    sorted_input = input_data[input_data[:, i].argsort()]

    '''getting the sorted label column'''
    label_col = sorted_input[:, 3]

    '''counting number of M and W'''
    m_count, w_count = len(label_col[label_col == 0]), len(label_col[label_col == 1])

    '''Calculating root entropy'''
    root_entropy = cal_entropy([m_count, w_count])

    '''Calculating the size of training data'''
    training_data_size = sorted_input.shape[0]

    gain_ratio_array = np.empty((training_data_size - 1, 5))

    for j in range(1, training_data_size):
        output = cal_gain_ration_for_data_set(sorted_input, root_entropy, i, j)
        if output is not None:
            gain_ratio_array[j - 1] = output
        # pass

    p = 0

    '''Finding the index of the max Gain Ratio'''
    max_index = gain_ratio_array[:, 0].argmax()

    '''Getting the max gain_ratio'''
    best_gain_ratio_feature = gain_ratio_array[max_index, 0]

    '''Getting the threshold which gave the Max Gain Ratio'''
    best_threshold_feature = gain_ratio_array[max_index, 1]

    '''Getting the point at which split occurred'''
    count = gain_ratio_array[max_index, 2]

    return best_gain_ratio_feature, best_threshold_feature, count, max_index


def get_next_node(feature_indexes, input_data, feature_dict, depth):


    best_gain_ratio_across_features = []
    best_threshold_across_features = []
    count_array = []

    for i in feature_indexes:
        best_gain_ratio_feature, best_threshold_feature, count, max_index = \
            choose_feature_and_threshold(input_data, i)

        '''Storing Best Gain Ratio for this feature'''
        best_gain_ratio_across_features.append(best_gain_ratio_feature)
        '''Storing Best Threshold for this feature'''
        best_threshold_across_features.append(best_threshold_feature)

        count_array.append(count)
        pass

    best_gain_ratio_index = best_gain_ratio_across_features.index(max(best_gain_ratio_across_features))
    best_gain_ratio = best_gain_ratio_across_features[best_gain_ratio_index]
    best_threshold = best_threshold_across_features[best_gain_ratio_index]
    split_point = count_array[best_gain_ratio_index]

    i = best_gain_ratio_index
    sorted_input = input_data[input_data[:, i].argsort()]
    split_point = int(split_point)
    part1, part2 = sorted_input[:split_point, :], sorted_input[split_point:, :]

    newNode = TreeNode()
    newNode.add_stats(feature_name=feature_dict[best_gain_ratio_index],
                          feature_index=best_gain_ratio_index, best_gain_ratio=best_gain_ratio,
                          feature_threshold=best_threshold, split_point=split_point)
    newNode.depth = depth + 1

    part1_label, part2_label = part1[:, 3], part2[:, 3]
    '''Counting M and W in each split'''
    left_m, left_w = len(part1_label[part1_label == 0]), len(part1_label[part1_label == 1])
    right_m, right_w = len(part2_label[part2_label == 0]), len(part2_label[part2_label == 1])

    top = 0

    if left_m == len(part1_label):
        leafNode = TreeNode()
        leafNode.leaf = True
        leafNode.decision = 'M'
        leafNode.depth = newNode.depth + 1
        newNode.leftNode = leafNode

    if left_w == len(part1_label):
        leafNode = TreeNode()
        leafNode.leaf = True
        leafNode.decision = 'W'
        leafNode.depth = newNode.depth + 1
        newNode.leftNode = leafNode

    if right_m == len(part2_label):
        leafNode = TreeNode()
        leafNode.leaf = True
        leafNode.decision = 'M'
        leafNode.depth = newNode.depth + 1
        newNode.rightNode = leafNode

    if right_w == len(part2_label):
        leafNode = TreeNode()
        leafNode.leaf = True
        leafNode.decision = 'W'
        leafNode.depth = newNode.depth + 1
        newNode.rightNode = leafNode

    if newNode.leftNode is None:
        newNode.leftNode = get_next_node(feature_indexes, part1, feature_dict, newNode.depth)

    if newNode.rightNode is None:
        newNode.rightNode = get_next_node(feature_indexes, part2, feature_dict, newNode.depth)

    return newNode
    # return best_gain_ratio_index, best_gain_ratio, best_threshold, split_point


def main():
    # filename = 'datasets/Q1_b_training_data.txt'
    filename = '../../data/Q1_C_training.txt'
    # height_idx, weight_idx, age_idx, label_idx = 0, 1, 2, 3
    feature_indexes = [0, 1, 2]

    feature_dict = {
        0: 'height',
        1: 'weight',
        2: 'age'
    }

    rootNode: TreeNode

    input_data_from_file = fetch_data(filename)

    td = np.array(input_data_from_file)
    input_data = add_numeric_labels(td)
    depth = 8

    '''Creating an empty Tree Node'''
    # root = TreeNode()
    # root.dataset = input_data

    # current = root

    starting_depth = -1

    root = get_next_node(feature_indexes, input_data, feature_dict, starting_depth)

    '''
    1. Add clssificaiton decision on each node based on majority
    2. Add depth property to each node 
    '''

    print('HI')

    # while current.stats is None and current.dataset is not

    # for d in range(depth):
    #     print('d=', d)
    #
    #     if root.stats is None and root.dataset is not None:

            # best_gain_ratio_index, best_gain_ratio, best_threshold, split_point \
            #     = get_next_node(feature_indexes, input_data, feature_dict)

            # pass
            # i = best_gain_ratio_index
            # sorted_input = input_data[input_data[:, i].argsort()]
            # split_point = int(split_point)
            # p1, p2 = sorted_input[:split_point, :], sorted_input[split_point:, :]
            #
            # first = TreeNode()
            # first.add_stats(feature_name=feature_dict[best_gain_ratio_index],
            #               feature_index=best_gain_ratio_index, best_gain_ratio=best_gain_ratio,
            #               feature_threshold=best_threshold, split_point=split_point)
            #
            # left, right = TreeNode(), TreeNode()
            # left.dataset, right.dataset = p1, p2
            #
            # first.leftNode = left
            # first.rightNode = right
            #
            # root = first



            # root = TreeNode(feature_name=feature_dict[best_gain_ratio_index],
            #               feature_index=best_gain_ratio_index, best_gain_ratio=best_gain_ratio,
            #               threshold=best_threshold, split_point=split_point)
            #
            # best_gain_ratio_index, best_gain_ratio, best_threshold, split_point \
            #     = get_next_node(feature_indexes, p1)
            #
            # i = best_gain_ratio_index
            # sorted_p1 = p1[p1[:, i].argsort()]
            # split_point = int(split_point)
            #
            # q1, q2 = sorted_p1[:split_point, :], sorted_p1[split_point:, :]
            #
            # rootNode.add_child(TreeNode(feature_name=feature_dict[best_gain_ratio_index],
            #               feature_index=best_gain_ratio_index,
            #               threshold=best_threshold, split_point=split_point))
            #
            # h = 0
            # pass


        # pass


if __name__ == "__main__":
    main()