import unittest
from python.Q1_B import *
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


if __name__ == '__main__':
    unittest.main()
