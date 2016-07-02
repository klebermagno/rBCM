"""Test that the rBCM produces predictions comparable to
a full gpr model in situations in which it reasonably should.

This module contains both visual and normal tests. To view the visual
tests output see the `tests/visuals` directory where they were placed.
"""
from __future__ import division
import numpy as np
import time

from sklearn.utils.estimator_checks import check_estimator
from sklearn.gaussian_process.gpr import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from rBCM.rbcm import RobustBayesianCommitteeMachineRegressor as RBCM
from tests.gpr_comparison_plots import compare_1d_plots


def f1(x):
    x = x / 1000
    return 4 * np.abs(np.sin(x)) + x


def f2(x):
    x = x / 1000
    return 10 * np.exp(-x) + x * np.abs(np.sin(x))


def add_noise(y, noise_level=1):
    return y + np.random.normal(0, noise_level, y.shape[0])


# def test_is_valid_estimator():
#     """Test that the class implements the BaseEstimator interface."""
#     check_estimator(RBCM)


def get_gpr_model_predictions(X, y, X_prime):
    kernel = C(1.0) * RBF(1.0) + WhiteKernel(1.0)
    gpr = GPR(kernel=kernel, n_restarts_optimizer=2)
    print("starting gpr fitting")
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, 1))
    X_prime = scaler.transform(X_prime.reshape(-1, 1))
    gpr.fit(X, y)
    print("starting gpr predicting")
    preds, sigma = gpr.predict(X_prime, return_std=True)
    return preds, sigma


def get_rbcm_model_predictions(X, y, X_prime):
    kernel = C(1.0) * RBF(1.0) + WhiteKernel(1.0)
    machine = RBCM(kernel=kernel, n_restarts_optimizer=2, standardize_X=True, points_per_expert=512, locality=False)
    print("starting rbcm fitting")
    machine.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    expert_sets = machine.sample_sets
    print("starting rbcm predicting")
    preds, var = machine.predict(X_prime.reshape(-1, 1), return_var=True, in_parallel=True)
    return preds, np.sqrt(var), expert_sets


def test_visual():
    X_1 = np.linspace(10, 4000, 1024, dtype=np.float64)
    X1 = np.linspace(10, 4000, 1024, dtype=np.float64)
    y1 = f1(X1)
    y1 = add_noise(y1)

    t = time.time()
    gpr_fit1, gpr_sigma = get_gpr_model_predictions(X1, y1, X_1)
    gpr_time = float(int((time.time() - t) * 1000)) / 1000

    # Sleep to cool off between runs while I go get pizza
    # time.sleep(120)

    t = time.time()
    rbcm_fit1, rbcm_sigma, expert_sets = get_rbcm_model_predictions(X1, y1, X_1)
    rbcm_time = float(int((time.time() - t) * 1000)) / 1000
    compare_1d_plots("linear_noisy", X1.ravel(), y1.ravel(), X_1.ravel(),
                     gpr_fit1.ravel(), gpr_sigma.ravel(), gpr_time,
                     rbcm_fit1.ravel(), rbcm_sigma.ravel(), rbcm_time, expert_sets)
