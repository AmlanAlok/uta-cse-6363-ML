import numpy as np

def read_data_ass1(filename):
    X = []
    y = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            clean_line = line.replace('((','').replace('(','').replace('\n','').replace(')','').strip().split(', ')
            X.append([float(x) for x in clean_line[:-1]])
            y.append(clean_line[-1])
        return np.array(X),np.array(y)