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


def get_feature_vector(x_data, k, d):

    ones = np.ones((x_data.shape[0], 1))
    p = np.concatenate((ones, x_data), axis=1)

    i = 1

    while i <= d:

        ikx = x_data*i*k

        sin_col = np.sin(ikx)
        p = np.concatenate((p, sin_col), axis=1)

        cos_col = np.cos(ikx)
        p = np.concatenate((p, cos_col), axis=1)

        i += 1

    return p

'''Uses all Data'''
def separate_input_output(input_data):

    td = np.array(input_data, dtype='float64')

    x_data_points = td[:, 0]
    x_data = x_data_points.reshape(x_data_points.shape[0], 1)

    y_data_points = td[:, 1]
    y_data = y_data_points.reshape(y_data_points.shape[0], 1)

    return x_data, y_data

'''Uses all Data'''
def separate_input_output_limit(input_data, limit=129):

    td = np.array(input_data, dtype='float64')

    x_data_points = td[:limit, 0]
    x_data = x_data_points.reshape(x_data_points.shape[0], 1)

    y_data_points = td[:limit, 1]
    y_data = y_data_points.reshape(y_data_points.shape[0], 1)

    return x_data, y_data


def train_linear_regression_model(k, d, size):

    filename = '../datasets/Q1_b_training_data.txt'
    input_data = fetch_data(filename)

    # x_data, y_data = separate_input_output(input_data)
    x_data, y_data = separate_input_output_limit(input_data, size)

    feature_matrix = get_feature_vector(x_data, k, d)

    ''' moore-penrose pseudoinverse numpy '''
    pseudo_inv = np.linalg.pinv(feature_matrix)

    parameter_matrix = np.matmul(pseudo_inv, y_data)

    return parameter_matrix


def prediction(x, parameter_matrix, k, d):

    x_data = x.reshape(x.shape[0], 1)
    feature_vector = get_feature_vector(x_data, k, d)
    prediction = np.matmul(feature_vector, parameter_matrix)

    return prediction


def error_calculation_test_data(parameter_matrix, k, d):

    filename = '../datasets/Q1_c_test_data.txt'
    test_data = fetch_data(filename)

    x_data, y_true = separate_input_output(test_data)

    y_prediction = prediction(x_data, parameter_matrix, k, d)

    ''' calculating mean square error '''
    mse = np.square(np.subtract(y_true, y_prediction)).mean()

    return mse


def main():

    print('program started')
    k = 4
    max_d = 6
    x = np.linspace(-3, 3, 1000)

    training_size = [129, 20, 10, 5]   # max = 129

    for size in training_size:

        line_names = []

        for d in range(max_d+1):

            parameter_matrix = train_linear_regression_model(k, d, size)
            '''You can save np array using this function'''
            # np.savetxt('./Q1/parameter-d-'+str(d)+'.csv', parameter_matrix, delimiter=',')

            ''' plotting graph '''
            plt.plot(x, prediction(x, parameter_matrix, k, d))

            ''' Error Calculation '''
            mse = error_calculation_test_data(parameter_matrix, k, d)

            line_names.append('d='+str(d)+', MSE='+str(mse))

        # Reading the csv into an array
        # firstarray = np.genfromtxt("firstarray.csv", delimiter=",")
        plt.title('Training Data Size ='+str(size))
        plt.legend(line_names)
        plt.savefig('./Q1/Q1-size-'+str(size))
        # plt.show()
        # plt.close()

    print('program end')
    pass


if __name__ == "__main__":
    main()


