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


# def get_distance(input_data, test_input):
#
#     a = [None]*len(input_data)
#     i = 0
#
#     for dp in input_data:
#         cartesian_distance = math.sqrt(
#             pow(dp['input']['height']-test_input['input']['height'], 2) +
#             pow(dp['input']['weight']-test_input['input']['weight'],2) +
#             pow(dp['input']['age']-test_input['input']['age'], 2))
#
#         a[i] = {
#             'index': dp['index'],
#             'cartesian_distance': cartesian_distance,
#             'output': dp['output']
#         }
#
#         i += 1
#
#     sorted_a = sorted(a, key=lambda d: d['cartesian_distance'])
#
#     # print(sorted_a)
#
#     return sorted_a


def calc_distance_from_given_point(x_supervised, y_supervised, dp):

    xy_supervised = np.concatenate((x_supervised, y_supervised), axis=1)
    supervised_size = xy_supervised.shape[0]
    dist_mat = np.full((supervised_size, 3), float('inf'))
    dist_index = 1
    inverse_dist_index = 2
    label_index = 6

    """
    dist_mat => 0 = dp index, 1 => distance from test dp, 2 => inverse distance from test dp
    """

    for i in range(supervised_size):
        dist_mat[i][0] = i
        d = np.linalg.norm(x_supervised[i] - dp)
        dist_mat[i][dist_index] = d
        dist_mat[i][inverse_dist_index] = 1/d


    xy_supervised_dist = np.concatenate((dist_mat, xy_supervised), axis=1)
    xy_supervised_dist = xy_supervised_dist[xy_supervised_dist[:, dist_index].argsort()]
    pass
    return xy_supervised_dist, inverse_dist_index, label_index


def semi_supervised_calc_distance_from_given_point(x_supervised, y_supervised, dp, k_knn):

    xy_supervised = np.concatenate((x_supervised[:, 1:], y_supervised[:, 1:]), axis=1)
    supervised_size = xy_supervised.shape[0]
    dist_mat = np.full((supervised_size, 3), float('inf'))
    dist_index = 1
    inverse_dist_index = 2
    label_index = 6
    x_supervised_dp = x_supervised[:, 1:]
    for i in range(supervised_size):
        dist_mat[i][0] = i
        d = np.linalg.norm(x_supervised_dp[i] - dp)
        dist_mat[i][dist_index] = d
        dist_mat[i][inverse_dist_index] = 1/d

    xy_supervised_dist = np.concatenate((dist_mat, xy_supervised), axis=1)
    xy_supervised_dist = xy_supervised_dist[xy_supervised_dist[:, inverse_dist_index].argsort()[::-1][:k_knn]]
    pass
    return xy_supervised_dist, inverse_dist_index, label_index


'''Q2 (a)'''


def weighted_voting(closest_k_points, inverse_dist_index, label_index):

    size = closest_k_points.shape[0]
    w_vote, m_vote = 0, 0

    for i in range(size):

        if closest_k_points[i][label_index] == 1:
            w_vote += closest_k_points[i][inverse_dist_index]
        if closest_k_points[i][label_index] == 0:
            m_vote += closest_k_points[i][inverse_dist_index]

    if w_vote > m_vote:
        final_label = 1
        vote_diff = w_vote - m_vote
    else:
        final_label = 0
        vote_diff = m_vote - w_vote

    return final_label, vote_diff


def check_accuracy(prediction, y_label):

    size = prediction.shape[0]
    correct_predictions = np.count_nonzero(prediction == y_label)   # gives the number of correct matches

    accuracy = correct_predictions/size
    return accuracy


def only_supervised_approach(x_supervised, y_supervised, x_unsupervised, y_unsupervised, k_knn):

    x_unsupervised_size = x_unsupervised.shape[0]
    # vote_diff = np.full((x_unsupervised_size, 1), -1)
    prediction = np.full((x_unsupervised_size, 1), -1)

    for i in range(x_unsupervised_size):
        xy_supervised_dist, inverse_dist_index, label_index = \
            calc_distance_from_given_point(x_supervised, y_supervised, x_unsupervised[i])
        '''Choosing the k closest points'''
        closest_k_points = xy_supervised_dist[:k_knn, :]
        final_label, voting_diff = weighted_voting(closest_k_points, inverse_dist_index, label_index)
        prediction[i][0] = final_label
        pass

    accuracy = check_accuracy(prediction, y_unsupervised)

    return accuracy


def add_index(x):

    size = x.shape[0]
    s_array = np.full((size, 1), -1)

    for i in range(size):
        s_array[i][0] = i

    joined = np.concatenate((s_array, x), axis=1)

    return joined


def semi_supervised_learning(x_supervised, y_supervised, x_unsupervised, y_unsupervised, k_knn, k):

    x_supervised = add_index(x_supervised)
    y_supervised = add_index(y_supervised)
    x_unsupervised = add_index(x_unsupervised)

    current_x_unsupervised_size = x_unsupervised.shape[0]
    current_x_unsupervised = x_unsupervised
    current_x_supervised = x_supervised
    current_y_supervised = y_supervised

    while current_x_unsupervised_size > 0:

        vote_diff_and_pred = np.full((current_x_unsupervised_size, 2), -1.0)
        vote_diff_index = 5
        current_x_unsupervised_dp = current_x_unsupervised[:, 1:]

        for j in range(current_x_unsupervised_size):

            closest_k_points, inverse_dist_index, label_index = \
                semi_supervised_calc_distance_from_given_point(current_x_supervised, current_y_supervised, current_x_unsupervised_dp[j], k_knn)

            final_label, voting_diff = weighted_voting(closest_k_points, inverse_dist_index, label_index)

            vote_diff_and_pred[j][0] = final_label
            vote_diff_and_pred[j][1] = voting_diff

        '''Adding the prediction and vote difference to all unsupervised data'''
        combine_with_unsupervised = np.concatenate((current_x_unsupervised, vote_diff_and_pred), axis=1)
        '''Choosing top k points with max vote_diff'''
        chosen_points = combine_with_unsupervised[combine_with_unsupervised[:, vote_diff_index].argsort()[::-1][:k]]

        '''Updating supervised datasets'''
        current_x_supervised = np.concatenate((current_x_supervised, chosen_points[:, :4]))

        p = np.array([chosen_points[:, 0]]).transpose()
        q = np.array([chosen_points[:, 4]]).transpose()

        y_record_array = np.concatenate((p,q), axis=1)

        current_y_supervised = np.concatenate((current_y_supervised, y_record_array))

        '''deleting records from current_x_unsupervised which were moved to the supervised set'''
        for t in range(chosen_points.shape[0]):
            current_x_unsupervised = current_x_unsupervised[current_x_unsupervised[:, 0] != chosen_points[t][0]]

        current_x_unsupervised_size = current_x_unsupervised.shape[0]

    '''collecting the unsupervised data predictions'''
    unsupervised_y_prediction = current_y_supervised[20:, :]
    unsupervised_y_prediction = unsupervised_y_prediction[unsupervised_y_prediction[:, 0].argsort()]

    prediction = np.array([unsupervised_y_prediction[:, 1]]).transpose()
    accuracy = check_accuracy(prediction, y_unsupervised)

    pass

    return accuracy


'''Q2: Self-Training'''


def main():

    partition = 20
    k_knn = 5

    '''number of data items that are being added to the labeled data set in each iteration'''
    k_array = [1, 5, 10, 20, 25, 50, 100]
    dict_ans = {}
    file_path = '../data/120_data_points.txt'
    training_input_data = get_input_data(file_path=file_path)

    """Splitting input data into (X, Y)"""
    input_data, y_label = split_data(training_input_data)

    '''Splitting the input data into supervised and unsupervised'''
    x_supervised, x_unsupervised = partition_input_data(input_data, partition)
    y_supervised, y_unsupervised = partition_label_data(y_label, partition)

    supervised_accuracy = only_supervised_approach(x_supervised, y_supervised, x_unsupervised, y_unsupervised, k_knn)

    for k in k_array:
        accuracy = semi_supervised_learning(x_supervised, y_supervised, x_unsupervised, y_unsupervised, k_knn, k)
        dict_ans[k] = accuracy
        pass

    dict_ans['supervised_accuracy'] = supervised_accuracy
    pass
    print(dict_ans)
    print('Finish')


if __name__ == "__main__":
    main()