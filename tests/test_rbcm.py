"""Functional unit tests for the RBCM class."""

import rBCM

import pytest
import numpy as np


class RegressionFixture(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y


class TestRBCM(object):
    @pytest.fixture
    def empty_data(self):
        X = np.array([])
        y = np.array([])
        return RegressionFixture(X, y)

    @pytest.fixture
    def single_point_1d_numpy_arrays(self):
        X = np.array([1])
        y = np.array([2])
        return RegressionFixture(X, y)

    @pytest.fixture
    def single_point_1d_py_lists(self):
        X = [1]
        y = [2]
        return RegressionFixture(X, y)

    @pytest.fixture
    def two_point_1d_data(self):
        X = np.array([[1], [2]])
        y = np.array([[5], [3]])
        return RegressionFixture(X, y)

    @pytest.fixture
    def single_point_2d_data(self):
        X = np.array([2, 3])
        y = np.array([5, 6])
        return RegressionFixture(X, y)

    @pytest.fixture
    def two_point_2d_data(self):
        X = np.array([[1, 2], [2, 3]])
        y = np.array([[5, 6], [6, 7]])
        return RegressionFixture(X, y)

    @pytest.fixture()
    def generic_small_data(self):
        X = np.array([[1, 2, 3, 4, 5],
                      [2, 3, 4, 5, 6],
                      [3, 4, 5, 6, 7],
                      [4, 5, 6, 7, 8]])
        y = np.array([[5, 6],
                      [6, 7],
                      [7, 8],
                      [8, 9]])
        return RegressionFixture(X, y)

    @pytest.fixture()
    def random_data(self):
        X = np.random.rand(100, 5)
        y = np.random.rand(100, 2)
        return RegressionFixture(X, y)

    def fitting_runner(self, regression_data):
        machine = rBCM.RBCM()
        machine.fit(regression_data.X, regression_data.y)
        return machine

    def test_instantiation(self):
        rBCM.RBCM()

    # ensure empty input is not accepted
    def test_empty_fit(self, empty_data):
        with pytest.raises(ValueError):
            self.fitting_runner(empty_data)

    # ensure python lists are not accepted
    def test_single_point_py_list_fit(self, single_point_1d_py_lists):
        with pytest.raises(ValueError):
            self.fitting_runner(single_point_1d_py_lists)

    def test_single_point_np_arr_fit(self, single_point_1d_numpy_arrays):
        self.fitting_runner(single_point_1d_numpy_arrays)

    def test_single_point_2d_fit(self, single_point_2d_data):
        self.fitting_runner(single_point_2d_data)

    def test_two_point_2d_fit(self, two_point_2d_data):
        self.fitting_runner(two_point_2d_data)

    def test_two_point_1d_fit(self, two_point_1d_data):
        self.fitting_runner(two_point_1d_data)

    def test_generic_small_data_fit(self, generic_small_data):
        self.fitting_runner(generic_small_data)

    def test_random_data_fit(self, random_data):
        self.fitting_runner(random_data)
