import numpy as np
import inspect
import math
# file_data = np.loadtxt('data/Q2_A.txt')
# print(file_data[0])


def clean_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')


def fetch_data(filename):
    print('inside func '+ inspect.stack()[0][3])

    with open(filename, 'r') as f:
        input_data = f.readlines()
        print(type(input_data ))
        print(len(input_data ))

        clean_input = list(map(clean_data, input_data))
        f.close()

    # return input_data
    return clean_input


def change_data_structure(input_data):

    print('inside func ' + inspect.stack()[0][3])

    data_list = [None] * len(input_data)
    i = 0

    for record in input_data:
        data_list[i] = {
            'index': i,
            'input': {
                'height': float(record[0]),
                'weight': float(record[1]),
                'age': int(record[2])
            },
            'output': record[3]
        }

        i += 1

    print('Converted each data point to dictionary format')

    return data_list


def get_knn(input_data, test_input, k):

    print('inside func ' + inspect.stack()[0][3])

    a = [None]*len(input_data)
    i = 0

    for dp in input_data:
        cartesian_distance = math.sqrt(
            pow(dp['input']['height']-test_input['input']['height'], 2) +
            pow(dp['input']['weight']-test_input['input']['weight'],2) +
            pow(dp['input']['age']-test_input['input']['age'], 2))

        a[i] = {
            'index': dp['index'],
            'cartesian_distance': cartesian_distance,
            'output': dp['output']
        }

        i += 1

    sorted_a = sorted(a, key=lambda d:d['cartesian_distance'], reverse=True)

    print(sorted_a)

    return sorted_a[:k]


filename = 'data/Q2_A.txt'

input_data = fetch_data(filename)
input_data = change_data_structure(input_data)

# user_input = input('Enter test data point:')
# print(user_input)
user_input = '( 1.7512428413306, 73.58553700624, 34)'
user_input = clean_data(user_input)
print(user_input)

k = 2

test_input_record = {
    'input': {
        'height': float(user_input[0]),
        'weight': float(user_input[1]),
        'age': int(user_input[2])
    },
    'knn': [],
    'prediction': ''
}


knn_array = get_knn(input_data, test_input_record, k)

W_count = 0
M_count = 0

for dic in knn_array:
    if dic['output'] == 'W':
        W_count += 1
    if dic['output'] == 'M':
        M_count += 1

W_prob = W_count/k
M_prob = M_count/k

print('W prob =', W_prob)
print('M_prob =', M_prob)

if W_count > M_count:
    test_input_record['prediction'] = 'W'
elif M_count > W_count:
    test_input_record['prediction'] = 'M'
else:
    test_input_record['prediction'] = '50-50'

print('prediction =', test_input_record['prediction'])


print('End')


# x = input_data[0].split(',')
# print(x)


