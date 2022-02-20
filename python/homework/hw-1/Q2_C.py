from Q2_my_functions import *

filename = 'data/Q2_C.txt'

input_data = fetch_data(filename)
k_list = [1, 3, 5]

result = leave_one_out(input_data, k_list)

result = result_accuracy(result, k_list)

print('END')

'''
The Accuracy values are as follows:
k = 1, 68.33 %
k = 3, 73.33 %
k = 5, 75.00 %

Hence, best performance is for k = 5  
'''