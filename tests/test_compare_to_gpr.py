"""Test that the rBCM produces predictions comparable to
a full gpr model in situations in which it reasonably should.
"""
import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from rBCM import RobustBayesianCommitteeMachineRegressor as rbcm


def f1(x):
    return x * np.sin(x)


def f2(x):
    return (x**2) * np.sin(x)

X = np.atleast_2d(np.arange(100)).T
X = np.atleast_2d(np.arange(100000)).T


def test_is_valid_estimator():
    """Test that the class implements the BaseEstimator interface."""
    check_estimator(rbcm)


def test_1d_small():
    pass
