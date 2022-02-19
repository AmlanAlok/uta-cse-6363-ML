from my_functions import *

filename = 'data/Q2_A.txt'

input_data = fetch_data(filename)
# input_data = change_data_structure(input_data)

# user_input = input('Enter test data point:')
# print(user_input)
user_input = '( 1.7512428413306, 73.58553700624, 34)'
user_input = clean_data(user_input)
print(user_input)

k_list = [1, 3, 5]

test_input_record = {
    'input': {
        'height': float(user_input[0]),
        'weight': float(user_input[1]),
        'age': int(user_input[2])
    },
    'knn': k_list,
    'output': []
}


distance_array = get_distance(input_data, test_input_record)

test_input_record = make_prediction(k_list, distance_array, test_input_record)

print('End')


# x = input_data[0].split(',')
# print(x)


