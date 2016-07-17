"""Test that the rBCM produces predictions comparable to
a full gpr model in situations in which it reasonably should.

This module contains both visual and normal tests. To view the visual
tests output see the `tests/visuals` directory where they were placed.
"""
from __future__ import division
import numpy as np
import time

from sklearn.gaussian_process.gpr import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from rBCM.rbcm import RobustBayesianCommitteeMachineRegressor as RBCM
from tests.gpr_comparison_plots import compare_1d_plots
from tests.data_generators import f2, add_noise


# def test_is_valid_estimator():
#     """Test that the class implements the BaseEstimator interface."""
#     check_estimator(RBCM)


def get_gpr_model_predictions(X, y, X_prime, n_restarts_optimizer=0):
    """Convenience function to ensure the predictions made in this module
    for both the gpr and rbcm are the same."""
    kernel = C(1.0) * RBF(1.0) + WhiteKernel(1e-12)
    gpr = GPR(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, 1))
    X_prime = scaler.transform(X_prime.reshape(-1, 1))
    gpr.fit(X, y)
    preds, sigma = gpr.predict(X_prime, return_std=True)
    return preds, sigma


def get_rbcm_model_predictions(X, y, X_prime, ppe=512, locality=False, n_restarts_optimizer=0):
    """Convenience function to ensure the predictions made in this module
    for both the gpr and rbcm are the same."""
    kernel = C(1.0) * RBF(1.0) + WhiteKernel(1e-12)
    machine = RBCM(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer,
                   standardize_X=True, points_per_expert=ppe, locality=locality)
    machine.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    expert_sets = machine.sample_sets
    preds, var = machine.predict(X_prime.reshape(-1, 1), return_var=True, in_parallel=True)
    return preds, np.sqrt(var), expert_sets


def test_visual_gpr_vs_rbcm():
    """Generates a plot on static html in the tests/visuals directory
    comparing the output from a full GPR and an rbcm model. This directory
    should be checked after running the tests to ensure everything looks
    normal."""
    data_scale = 1024 * 1.99
    ppe_scale = 512
    X_1 = np.linspace(0, 1000, 1024, dtype=np.float64)
    X1 = np.linspace(0, 1000, data_scale, dtype=np.float64)
    y1 = f2(X1)
    y1 = add_noise(y1, 0.2)

    t = time.time()
    gpr_fit, gpr_sigma = get_gpr_model_predictions(X1, y1, X_1)
    gpr_time = float(int((time.time() - t) * 1000)) / 1000

    t = time.time()
    rbcm_fit, rbcm_sigma, expert_sets = get_rbcm_model_predictions(X1, y1, X_1,
                                                                   ppe=ppe_scale,
                                                                   locality=False)
    rbcm_time = float(int((time.time() - t) * 1000)) / 1000
    compare_1d_plots("linear_noisy", X1.ravel(), y1.ravel(), X_1.ravel(),
                     gpr_fit.ravel(), gpr_sigma.ravel(), gpr_time,
                     rbcm_fit.ravel(), rbcm_sigma.ravel(), rbcm_time, expert_sets)


def test_degenerates_to_gpr():
    """Test that with small data and no given points per expert that the rBCM
    degenerates to a full GPR so as to not lose predictive accuracy. Testing
    permits no difference in output between the two."""
    X_1 = np.linspace(0, 100, 100, dtype=np.float64)
    X1 = np.linspace(0, 100, 512, dtype=np.float64)
    y1 = f2(X1)
    y1 = add_noise(y1, 0.2)

    gpr_fit, gpr_sigma = get_gpr_model_predictions(X1, y1, X_1, 0)
    rbcm_fit, rbcm_sigma, expert_sets = get_rbcm_model_predictions(X1, y1, X_1, ppe=None, locality=False, n_restarts_optimizer=0)
    # Since we degenerated to the same GPR, we should have identical values
    np.testing.assert_allclose(gpr_fit.ravel(), rbcm_fit.ravel(), rtol=1e-3)
    np.testing.assert_allclose(gpr_sigma.ravel(), rbcm_sigma.ravel(), rtol=1e-3)


def test_full_points_per_expert_is_gpr():
    """Checks that the results of the rBCM are equal to those of the full gpr
    when the rBCM should have degenerated to a single full GPR since it had
    it's points_per_expert parameter set to the full data set"""
    X_1 = np.linspace(0, 100, 100, dtype=np.float64)
    X1 = np.linspace(0, 100, 512, dtype=np.float64)
    y1 = f2(X1)
    y1 = add_noise(y1, 0.2)

    gpr_fit, gpr_sigma = get_gpr_model_predictions(X1, y1, X_1, 0)
    rbcm_fit, rbcm_sigma, expert_sets = get_rbcm_model_predictions(X1, y1, X_1, ppe=X1.shape[0], locality=False, n_restarts_optimizer=0)
    # Since we degenerated to the same GPR, we should have identical values
    np.testing.assert_allclose(gpr_fit.ravel(), rbcm_fit.ravel(), rtol=1e-3)
    np.testing.assert_allclose(gpr_sigma.ravel(), rbcm_sigma.ravel(), rtol=1e-3)
