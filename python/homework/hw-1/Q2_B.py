from Q2_my_functions import *

print('------------------------------')
filename = input('Enter filename from the data directory\n'
                 'Default file is Q2_A.txt\n'
                 'You can add your own data in a txt file and enter filename here.\n'
                 'Or you can hit enter to proceed with the default dataset:')

if filename == '':
    filename = 'Q2_A.txt'
file_path = 'python/homework/hw-1/data/' + filename

input_data = fetch_data(file_path)

print('------------------------------')
user_input = input('Enter test data point. \n'
                   'Default = ( 1.7512428413306, 73.58553700624, 34) \n'
                   'Enter data of your choice or just hit enter. The program will execute the default one shown above :')
if user_input == '':
    user_input = '( 1.7512428413306, 73.58553700624, 34)'
print('You entered ->', user_input)
# user_input = '( 1.7512428413306, 73.58553700624, 34)'
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


