from Q2_my_functions import *
from Q3_my_functions import *

print('------------------------------')
filename = input('Enter filename from the data directory\n'
                 'Default file is Q2_C.txt\n'
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

test_input_record = {
    'input': {
        'height': float(user_input[0]),
        'weight': float(user_input[1]),
        'age': int(user_input[2])
    },
    'output': []
}

test_result = gaussian_naive_bayes(input_data, test_input_record)

print('Final prediction =', test_result['prediction'])

print('END')