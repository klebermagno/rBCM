"""This module contains tests for the validity of the combination of the
predictions and variances from many experts into one."""

import numpy as np
from tests.data_generators import f2, add_noise


# def test_visual():
#     data_scale = 1024 * 1.99
#     ppe_scale = 512
#     X_1 = np.linspace(0, 1000, 1024, dtype=np.float64)
#     X1 = np.linspace(0, 1000, data_scale, dtype=np.float64)
#     y1 = f2(X1)
#     y1 = add_noise(y1, 1)

#     gpr_fit, gpr_sigma = get_gpr_model_predictions(X1, y1, X_1)

#     rbcm_fit, rbcm_sigma, expert_sets = get_rbcm_model_predictions(X1, y1, X_1, ppe=ppe_scale, locality=False)
#     rbcm_time = float(int((time.time() - t) * 1000)) / 1000
#     compare_1d_plots("linear_noisy", X1.ravel(), y1.ravel(), X_1.ravel(),
#                      gpr_fit.ravel(), gpr_sigma.ravel(), gpr_time,
#                      rbcm_fit.ravel(), rbcm_sigma.ravel(), rbcm_time, expert_sets)
