import numpy as np
import matplotlib.pyplot as plt


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


'''Uses all Data'''
def separate_input_output(input_data, leave_pos):

    td = np.array(input_data)

    h_test = td[leave_pos, 0]
    w_test = td[leave_pos, 1]
    a_test = td[leave_pos, 2]

    height_data_points = np.concatenate((td[:leave_pos, 0], td[leave_pos+1:, 0]))
    height_data = height_data_points.reshape(height_data_points.shape[0], 1)
    height_data = height_data.astype('float64')

    weight_data_points = np.concatenate((td[:leave_pos, 1], td[leave_pos+1:, 1]))
    weight_data = weight_data_points.reshape(weight_data_points.shape[0], 1)
    weight_data = weight_data.astype('float64')

    age_data_points = np.concatenate((td[:leave_pos, 2], td[leave_pos+1:, 2]))
    age_data = age_data_points.reshape(age_data_points.shape[0], 1)
    age_data = age_data.astype('int64')

    # y_data_points = np.concatenate((td[:leave_pos, 3], td[leave_pos+1:, 3]))
    y_data_points = td[:, 3]
    y_data = y_data_points.reshape(y_data_points.shape[0], 1)
    y_data_01 = change_y_data(y_data)
    # y_data_01_leave_out = np.concatenate(y_data_01[:leave_pos], y_data_01[leave_pos+1:])
    y_true = y_data_01[leave_pos]
    y_data_01_leave_out = np.delete(y_data_01, leave_pos)

    return height_data, weight_data, age_data, y_data_01_leave_out, h_test, w_test, a_test, y_true


def get_feature_matrix(height_data, weight_data, age_data):

    return np.array([1, height_data, weight_data, age_data], dtype='float64').reshape(4, 1)


def get_random_parameter_matrix():
    return np.random.randn(4, 1)


def change_y_data(y_data):
    value_01 = np.unique(y_data, return_inverse=True)[1]
    return value_01


def sigmoid(x):
    return 1/(1+np.exp(-1*x))


def prediction(parameter_matrix, feature_matrix):

    linear_regression_output = np.matmul(np.transpose(parameter_matrix), feature_matrix)

    sigmoid_output = sigmoid(linear_regression_output)

    if sigmoid_output >= 0.5:
        return 1
    return 0


def train(alpha, iterations, height_data, weight_data, age_data, y_data_01):

    parameter_matrix = get_random_parameter_matrix()
    # y_data_01 = change_y_data(y_data)

    for k in range(iterations):

        error_array = []

        for i in range(height_data.shape[0]):

            feature_matrix = get_feature_matrix(height_data[i], weight_data[i], age_data[i])

            y_prediction = prediction(parameter_matrix, feature_matrix)

            err = y_prediction - y_data_01[i]
            error_array.append(err)

            partial_derivative = alpha * err * feature_matrix

            parameter_matrix = parameter_matrix - partial_derivative

        error_np = np.array(error_array)
        accuracy = 100 - (np.sum(np.square(error_np))/error_np.size)*100
        # print('Itr =', k, ' accuracy =', accuracy)

    # print(parameter_matrix)

    return parameter_matrix


def main():

    print('program started')
    alpha = 0.01
    iterations = 1

    filename = '../datasets/Q3_data.txt'
    input_data = fetch_data(filename)
    error_array = []

    for i in range(len(input_data)):

        height_data, weight_data, age_data, y_data_01, h_test, w_test, a_test, y_true = separate_input_output(input_data, i)  # put i here

        parameter_matrix = train(alpha, iterations, height_data, weight_data, age_data, y_data_01)

        feature_matrix = get_feature_matrix(h_test, w_test, a_test)
        # print('h,w,a =', h_test, w_test, a_test)
        y_prediction = prediction(parameter_matrix, feature_matrix)

        err = y_prediction - y_true
        error_array.append(err)

    error_np = np.array(error_array)
    accuracy = 100 - (np.sum(np.square(error_np)) / error_np.size) * 100
    print('Leave one out Accuracy = ', accuracy, ' %')
    print('program ended')


if __name__ == "__main__":
    main()


