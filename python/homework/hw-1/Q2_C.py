from Q2_B import *

filename = 'data/Q2_C.txt'

input_data = fetch_data(filename)
k_list = [1, 3, 5]

result = leave_one_out(input_data, k_list)

result = result_accuracy(result, k_list)

print('END')