import numpy as np
import math


class TreeNode:

    def __init__(self):
        self.stats = None
        self.leftNode = None
        self.rightNode = None
        self.leaf = False
        self.decision = None
        self.depth = None

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
    print('[M, W] =', x)
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

    print('\nChoosing threshold value = ', c)

    '''splitting data according to this threshold'''
    count = np.count_nonzero(sorted_input[:, i] <= c)

    '''splitting the label column'''
    p1, p2 = sorted_input[:count, 3], sorted_input[count:, 3]
    print('splitting data by <= threshold criteria')
    print('left part =', p1)
    print('right part =', p2)

    '''Counting M and W in each split'''
    left_m, left_w = len(p1[p1 == 0]), len(p1[p1 == 1])
    right_m, right_w = len(p2[p2 == 0]), len(p2[p2 == 1])

    '''Calculating Information Gain'''
    information_gain = gain(root_entropy, [left_m, left_w], [right_m, right_w])
    print('information_gain =', information_gain)

    '''Calculating Gain Ratio'''
    if len(p1) == 0 or len(p2) == 0:
        return None

    gain_ratio = cal_gain_ratio(information_gain, [len(p1), len(p2)])
    print('Gain ratio =', gain_ratio)

    return [gain_ratio, c, count, len(p1), len(p2)]


def choose_feature_and_threshold(input_data, i):

    '''sorting the input by feature value'''
    sorted_input = input_data[input_data[:, i].argsort()]

    print('Sorting the data by feature')
    print(sorted_input)

    '''getting the sorted label column'''
    label_col = sorted_input[:, 3]

    '''counting number of M and W'''
    m_count, w_count = len(label_col[label_col == 0]), len(label_col[label_col == 1])

    '''Calculating root entropy'''
    root_entropy = cal_entropy([m_count, w_count])
    print('\nroot_entropy =', root_entropy)

    '''Calculating the size of training data'''
    training_data_size = sorted_input.shape[0]

    gain_ratio_array = np.empty((training_data_size - 1, 5))

    for j in range(1, training_data_size):
        output = cal_gain_ration_for_data_set(sorted_input, root_entropy, i, j)
        if output is not None:
            gain_ratio_array[j - 1] = output

    p = 0

    '''Finding the index of the max Gain Ratio'''
    max_index = gain_ratio_array[:, 0].argmax()
    print('\nAll gain ratio\'s for diff thresholds')
    print('Gain Ratio   |Threshold')
    print(gain_ratio_array[:, :2])

    '''Getting the max gain_ratio'''
    best_gain_ratio_feature = gain_ratio_array[max_index, 0]

    '''Getting the threshold which gave the Max Gain Ratio'''
    best_threshold_feature = gain_ratio_array[max_index, 1]

    '''Getting the point at which split occurred'''
    count = gain_ratio_array[max_index, 2]

    return best_gain_ratio_feature, best_threshold_feature, count, max_index


def get_next_node(feature_indexes, input_data, feature_dict, depth, allowed_depth):


    best_gain_ratio_across_features = []
    best_threshold_across_features = []
    count_array = []

    for i in feature_indexes:

        print('\n------------------------------------------------------------------')
        print('Checking Gain Ratio for feature = ', feature_dict[i])
        best_gain_ratio_feature, best_threshold_feature, count, max_index = \
            choose_feature_and_threshold(input_data, i)

        print('\nBest Gain Ratio =', best_gain_ratio_feature, ', with threshold =', best_threshold_feature)
        '''Storing Best Gain Ratio for this feature'''
        best_gain_ratio_across_features.append(best_gain_ratio_feature)
        '''Storing Best Threshold for this feature'''
        best_threshold_across_features.append(best_threshold_feature)

        count_array.append(count)

    print('Evaluating what to choose ...')
    print('Best gain ratio across Height, Weight and Age =', best_gain_ratio_across_features)
    best_gain_ratio_index = best_gain_ratio_across_features.index(max(best_gain_ratio_across_features))
    best_gain_ratio = best_gain_ratio_across_features[best_gain_ratio_index]
    best_threshold = best_threshold_across_features[best_gain_ratio_index]
    split_point = count_array[best_gain_ratio_index]

    print('Chosen highest Gain ratio =', best_gain_ratio, ', which has threshold =', best_threshold, ' of feature =', \
          feature_dict[best_gain_ratio_index])

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

    if newNode.depth > 0:
        man_count = left_m + right_m
        woman_count = left_w + right_w

        if man_count > woman_count:
            newNode.decision = 0
        else:
            newNode.decision = 1

    if newNode.depth < allowed_depth:
        if left_m == len(part1_label):
            leafNode = TreeNode()
            leafNode.leaf = True
            leafNode.decision = 0
            leafNode.depth = newNode.depth + 1
            newNode.leftNode = leafNode

        if left_w == len(part1_label):
            leafNode = TreeNode()
            leafNode.leaf = True
            leafNode.decision = 1
            leafNode.depth = newNode.depth + 1
            newNode.leftNode = leafNode

        if right_m == len(part2_label):
            leafNode = TreeNode()
            leafNode.leaf = True
            leafNode.decision = 0
            leafNode.depth = newNode.depth + 1
            newNode.rightNode = leafNode

        if right_w == len(part2_label):
            leafNode = TreeNode()
            leafNode.leaf = True
            leafNode.decision = 1
            leafNode.depth = newNode.depth + 1
            newNode.rightNode = leafNode

        if newNode.leftNode is None:
            newNode.leftNode = get_next_node(feature_indexes, part1, feature_dict, newNode.depth, allowed_depth)

        if newNode.rightNode is None:
            newNode.rightNode = get_next_node(feature_indexes, part2, feature_dict, newNode.depth, allowed_depth)

    return newNode


def get_prediction(record, root_node):

    current_node = root_node

    while current_node is not None:

        '''This indicates that we have reached a leaf node'''
        if current_node.leaf:
            return current_node.decision
        '''This means that we have reached the end of the tree but this is not a leaf node'''
        if current_node.leftNode is None and current_node.rightNode is None:
            return current_node.decision

        feature_index = current_node.stats['feature_index']
        feature_threshold = current_node.stats['feature_threshold']

        if record[feature_index] <= feature_threshold:
            current_node = current_node.leftNode
        else:
            current_node = current_node.rightNode


def run_accuracy_test_on_dataset(input_data, rootNode):

    dataset_size = input_data.shape[0]
    y_array = []

    for i in range(dataset_size):
        y_predict = get_prediction(input_data[i], rootNode)
        y_array.append(y_predict)

    y_predict = np.array(y_array)
    y_label = input_data[:, 3]

    mismatch = np.count_nonzero(y_label != y_predict)
    correct_pred = dataset_size - mismatch

    accuracy = (correct_pred/dataset_size)*100

    pass
    return accuracy


def question_2(training_input_data, test_input_data, feature_indexes, feature_dict, starting_depth):
    '''Q2'''

    bagging_array = [10, 50, 100]
    print('\nQ2 - BAGGING - Size =', bagging_array)
    chosen_depth = 4
    print('Chosen tree depth =', chosen_depth)

    test_dataset_size = test_input_data.shape[0]

    '''This is for the k different datasets and classifiers '''
    for bag_count in bagging_array:

        bagging_tree_dict = {}

        '''iterating through each k value'''
        for i in range(1, bag_count + 1):
            random_indices = np.random.choice(training_input_data.shape[0], training_input_data.shape[0])
            random_training_dataset = training_input_data[random_indices]

            rootNode = get_next_node(feature_indexes, random_training_dataset, feature_dict, starting_depth,
                                     chosen_depth)
            bagging_tree_dict[i] = rootNode

        y_array = []

        '''iterating through test data'''
        for j in range(test_dataset_size):

            classifier_outputs_array = []
            for k in range(1, bag_count + 1):
                y_predict = get_prediction(test_input_data[j], bagging_tree_dict[k])
                classifier_outputs_array.append(y_predict)

            classifier_outputs = np.array(classifier_outputs_array)

            m_count = len(classifier_outputs[classifier_outputs == 0])
            w_count = len(classifier_outputs[classifier_outputs == 1])

            if m_count > w_count:
                y_array.append(0)
            else:
                y_array.append(1)

        y_predict = np.array(y_array)
        y_label = test_input_data[:, 3]

        mismatch = np.count_nonzero(y_label != y_predict)
        correct_pred = test_dataset_size - mismatch

        accuracy = (correct_pred / test_dataset_size) * 100

        print('-----------------------------------------------------------')
        print('Bagging K =', bag_count, ', test data set accuracy =', accuracy)

    print('-----------------------------------------------------------')
    pass


def main():

    '''sol part B'''

    # filename = 'datasets/Q1_b_training_data.txt'
    training_filename = '../../data/Q1_A.txt'
    # training_filename = 'data/Q1_A.txt'

    feature_indexes = [0, 1, 2]
    feature_dict = {
        0: 'height',
        1: 'weight',
        2: 'age'
    }

    input_data_from_file = fetch_data(training_filename)
    td = np.array(input_data_from_file)
    training_input_data = add_numeric_labels(td)

    # depth_array = [1, 2, 3, 4, 5, 6, 7, 8]
    # depth_array = [4]
    depth_array = [2]
    starting_depth = -1
    decision_tree_dict = {}

    for allowed_depth in depth_array:
        rootNode = get_next_node(feature_indexes, training_input_data, feature_dict, starting_depth, allowed_depth)
        decision_tree_dict[allowed_depth] = rootNode

    '''sol Part C'''
    print('Finish')


if __name__ == "__main__":
    main()