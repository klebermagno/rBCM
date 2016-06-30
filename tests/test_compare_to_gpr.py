"""Test that the rBCM produces predictions comparable to
a full gpr model in situations in which it reasonably should.
"""
import numpy as np
from bokeh.plotting import figure, output_file, save

from sklearn.utils.estimator_checks import check_estimator
from sklearn.gaussian_process.gpr import GaussianProcessRegressor as GPR
from rBCM.rbcm import RobustBayesianCommitteeMachineRegressor as RBCM


n1 = 100
n2 = 1000


def f1(x):
    return (x**2) * np.abs(np.sin(x))


def f2(x):
    return x * np.abs(np.sin(x))


def add_noise(y):
    return y + np.random.normal(0, 0.2)

# Ground truth
X = np.linspace(0, 10, 1000)
y_1 = f1(X)
y_2 = f2(X)

# Data set 1
X1 = np.arange(n1, dtype=np.float64) / 10
y1 = f1(X1)
y1 = add_noise(y1)

print("Fitting first test data set")
machine1 = RBCM()
machine1.fit(X1.reshape(-1, 1), y1.reshape(-1, 1))
fit1 = machine1.predict(X.reshape(-1, 1))

# Data set 2
X2 = np.arange(n2, dtype=np.float64) / 100
y2 = f2(X2)
y2 = add_noise(y2)

print("Fitting second test data set")
machine2 = RBCM()
machine2.fit(X2.reshape(-1, 1), y2.reshape(-1, 1))
fit2 = machine2.predict(X.reshape(-1, 1))


# def test_is_valid_estimator():
#     """Test that the class implements the BaseEstimator interface."""
#     check_estimator(RBCM)


def test_small_data_visual():
    output_file("visuals/scatter1k.html", title="scatter")
    p = figure()
    p.scatter(X1, y1)
    p.line(X, fit1.ravel())
    save(p)


def test_large_data_visual():
    output_file("visuals/scatter10k.html", title="scatter")
    p = figure()
    p.scatter(X2, y2)
    p.line(X, fit2.ravel())
    save(p)
