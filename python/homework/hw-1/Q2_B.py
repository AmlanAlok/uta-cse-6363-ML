import numpy as np
import inspect
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


filename = 'data/Q2_A.txt'

input_data = fetch_data(filename)

print(type(input_data[0]))      # each line is in str dtype
print(input_data[0][0])
print(input_data[0][1])

print('---')
t = tuple(input_data)
print(t)


# x = input_data[0].split(',')
# print(x)


