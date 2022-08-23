import unittest
from python.Q1 import *
from sklearn.cluster import AgglomerativeClustering


class MyTestCase(unittest.TestCase):

    def test_single(self):
        file_path = '../data/120_data_points.txt'
        training_input_data = get_input_data(file_path=file_path)

        """Splitting input data into (X, Y)"""
        X, y_label = split_data(training_input_data)

        clustering = AgglomerativeClustering(linkage='single', n_clusters=2).fit(X)
        pass

    def test_complete(self):
        file_path = '../data/120_data_points.txt'
        training_input_data = get_input_data(file_path=file_path)

        """Splitting input data into (X, Y)"""
        X, y_label = split_data(training_input_data)

        clustering = AgglomerativeClustering(linkage='complete', n_clusters=4).fit(X)
        pass

    def test_average(self):
        file_path = '../data/120_data_points.txt'
        training_input_data = get_input_data(file_path=file_path)

        """Splitting input data into (X, Y)"""
        X, y_label = split_data(training_input_data)

        clustering = AgglomerativeClustering(linkage='average', n_clusters=2).fit(X)
        pass

    def test_hey(self):

        x = [2,3,4]
        y = [2,3,4]
        # a = [[1],[0],[1]]
        # b = [[1],[0],[1]]

        a = np.array([[1], [0], [1]])
        b = np.array([[1], [1], [1]])
        c = np.count_nonzero(x == y)
        print(c)
        print(a==b)
        d = np.count_nonzero(a == b)    # counts number of TRUEs
        print(d)

if __name__ == '__main__':
    unittest.main()
