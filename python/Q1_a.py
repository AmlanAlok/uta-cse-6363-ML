import numpy as np
import math as m
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


def train_linear_regression_model(k, d):

    filename = '../datasets/Q1_b_training_data.txt'
    input_data = fetch_data(filename)

    td = np.array(input_data, dtype='float64')
    del input_data

    x_data_points = td[:, 0]
    x_data = x_data_points.reshape(x_data_points.shape[0], 1)
    del x_data_points

    y_data_points = td[:, 1]
    y_data = y_data_points.reshape(y_data_points.shape[0], 1)
    del y_data_points

    feature_matrix = get_feature_vector(x_data, k, d)

    ''' moore-penrose pseudoinverse numpy '''
    pseudo_inv = np.linalg.pinv(feature_matrix)

    parameter_matrix = np.matmul(pseudo_inv, y_data)

    return parameter_matrix


def f(x, parameter_matrix, k, d):

    x_data = x.reshape(x.shape[0], 1)
    feature_vector = get_feature_vector(x_data, k, d)
    prediction = np.matmul(feature_vector, parameter_matrix)

    return prediction


def main():

    print('program started')
    k = 4
    max_d = 6
    x = np.linspace(-3, 3, 1000)
    line_names = []

    for d in range(max_d+1):
        parameter_matrix = train_linear_regression_model(k, d)
        np.savetxt('./Q1/parameter-d-'+str(d)+'.csv', parameter_matrix, delimiter=',')

        plt.plot(x, f(x, parameter_matrix, k, d))
        line_names.append('d='+str(d))

    # Reading the csv into an array
    # firstarray = np.genfromtxt("firstarray.csv", delimiter=",")
    plt.legend(line_names)
    plt.show()
    print('program end')
    pass


if __name__ == "__main__":
    main()


