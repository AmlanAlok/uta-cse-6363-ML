from Q2_my_functions import *
from Q3_my_functions import *

filename = 'data/Q2_C.txt'

input_data = fetch_data(filename)

result = leave_one_out(input_data)

result_no_age = leave_one_out(input_data, exclude_age=True)

print('Result with all 3 Height, Weight and Age =', result)
print('Result with all 3 Height and Weight =', result_no_age)