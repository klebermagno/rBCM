"""Test that the rBCM produces predictions comparable to
a full gpr model in situations in which it reasonably should.

This module contains both visual and normal tests. To view the visual
tests output see the `tests/visuals` directory where they were placed.
"""
from __future__ import division
import numpy as np

from bokeh.plotting import figure, output_file, save

from sklearn.utils.estimator_checks import check_estimator
from sklearn.gaussian_process.gpr import GaussianProcessRegressor as GPR

from rBCM.rbcm import RobustBayesianCommitteeMachineRegressor as RBCM


def f1(x):
    return np.power(x, 2) * np.abs(np.sin(x))


def f2(x):
    x = x / 5
    return x * np.abs(np.sin(x))


def add_noise(y, noise_level=0.1):
    return y + np.random.normal(0, noise_level, y.shape[0])


# def test_is_valid_estimator():
#     """Test that the class implements the BaseEstimator interface."""
#     check_estimator(RBCM)


def test_visual_comparison():
    X_1 = np.arange(1000, dtype=np.float64) / 100
    X1 = np.arange(100, dtype=np.float64) / 10
    y1 = f2(X1)
    y1 = add_noise(y1)

    gpr_fit1, sigma = get_gpr_model_predictions(X1, y1, X_1)
    rbcm_fit1, sigma = get_rbcm_model_predictions(X1, y1, X_1)
    create_comparison_plots("linear_noisy", X1.ravel(), y1.ravel(), X_1.ravel(),
                            gpr_fit1.ravel(), X_1.ravel(), rbcm_fit1.ravel())


def create_comparison_plots(output_filename, X, y, gpr_X, gpr_y, rbcm_X, rbcm_y):
    output_file("visuals/" + output_filename + "_rbcm.html", title="Comparison between RBCM and GPR")
    p1 = figure(title="RBCM Model")
    p1.scatter(X, y)
    p1.line(rbcm_X, rbcm_y)
    save(p1)

    output_file("visuals/" + output_filename + "_gpr.html", title="Comparison between RBCM and GPR")
    p2 = figure(title="GPR Model")
    p2.scatter(X, y)
    p2.line(gpr_X, gpr_y)
    save(p2)


def get_gpr_model_predictions(X, y, X_prime):
    gpr = GPR(n_restarts_optimizer=5, alpha=1e-3)
    gpr.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    preds, sigma = gpr.predict(X_prime.reshape(-1, 1), return_std=True)
    return preds, sigma


def get_rbcm_model_predictions(X, y, X_prime):
    machine = RBCM()
    machine.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    preds, var = machine.predict(X_prime.reshape(-1, 1), return_var=True)
    return preds, np.sqrt(var)
