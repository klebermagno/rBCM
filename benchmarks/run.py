"""Test that the rBCM produces predictions comparable to
a full gpr model in situations in which it reasonably should.

This module contains both visual and normal tests. To view the visual
tests output see the `tests/visuals` directory where they were placed.
"""
from __future__ import division
import sys
import numpy as np

from sklearn.gaussian_process.gpr import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from rBCM.rbcm import RobustBayesianCommitteeMachineRegressor as RBCM


def get_gpr_model_predictions(X, y, X_prime):
    kernel = C(1.0) * RBF(1.0) + WhiteKernel(1.0)
    gpr = GPR(kernel=kernel, n_restarts_optimizer=5)
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
    machine = RBCM(kernel=kernel, n_restarts_optimizer=5, standardize_X=True, points_per_expert=256)
    print("starting rbcm fitting")
    machine.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    print("starting rbcm predicting")
    preds, var = machine.predict(X_prime.reshape(-1, 1), return_var=True, in_parallel=True)
    return preds, np.sqrt(var)


def main(argv=None):
    pass

if __name__ == "__main__":
    main(sys.argv)
