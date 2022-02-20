from Q2_my_functions import *
from Q3_my_functions import *

filename = 'data/Q2_A.txt'
input_data = fetch_data(filename)

user_input = '( 1.7512428413306, 73.58553700624, 34)'
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

result = gaussian_naive_bayes(input_data, test_input_record)

print('END')