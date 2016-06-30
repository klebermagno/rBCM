"""Weighting and combination of sets of predictions into a single prediction."""

# Authors: Lucas Jere Kolstad <lucaskolstad@gmail.com>
#
# License: see LICENSE file

import time 

import cython
import numpy as np
cimport cython
cimport numpy as np

def differential_entropy_weighting(predictions, sigma, prior_std):
    """Weight the predictions of experts and reduce to a single prediction.

    This weighting function computes the beta uncertainty measure as:    
        The differential entropy between the prior predictive distribution and
        the posterior predictive distribution. Given as one half the
        difference between the log of the prior's variance and the log of the
        posterior's variance at the location of the prediction.

    See 'jmlr.org/proceedings/papers/v37/deisenroth15.pdf' page 5 for a
    description of this weighting in the context of an rBCM.

    Parameters:
    ----------
    predictions : array, shape = (n_locations, y_num_columns, num_experts)
        The values predicted by each expert

    sigma : array, shape = (n_locations, num_experts)
        The uncertainty of each expert at each location

    prior_std : float
        The standard deviation of the prior used to fit the GPRs
    
    Returns
    -------
    preds : array, shape = (n_locations, y_num_columns)
        Mean of predictive distribution at query points

    rbcm_var : array, shape = (n_locations, )
        Variance of predictive distribution at query points
    """
    var = np.power(sigma, 2)
    log_var = np.log(var)
    inv_var = 1 / var

    prior_var = np.power(prior_std, 2)
    log_prior_var = np.log(prior_var)

    # Compute beta weights, page 5 right hand column
    beta = np.zeros((sigma.shape[0], sigma.shape[1]))
    for j in range(predictions.shape[1]):
        beta[:, j] = (0.5) * (log_prior_var - log_var[:, j])

    return _combine(predictions, var, beta, prior_std)

    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef _combine(np.ndarray[np.float64_t, ndim=3] predictions,
              np.ndarray[np.float64_t, ndim=2] var,
              np.ndarray[np.float64_t, ndim=2] beta,
              double prior_var):
    """Calculate a single prediction from many with the given beta weights.

    Parameters
    -----------
    predictions : array-like, shape = (n_locations, n_features, n_experts)
        Values predicted by some sklearn predictor that offers var as well

    var : array-like, shape = (n_locations, n_experts)
        Variances corresponding to the predictions

    Returns
    -------
    predictions : array, shape = (n_locations, n_features)

    rbcm_var : array, shape = (n_locations)
    """
    # declaring loop variables
    cdef int i, j, k
    cdef double summation

    cdef np.ndarray[np.float64_t, ndim=2] inv_var = 1 / var
    cdef double inv_prior_var = 1 / prior_var

    # Compute Eq. 22
    beta_sums = np.zeros((predictions.shape[0], 1))
    left_term = np.zeros((predictions.shape[0], 1))
    right_term = np.zeros((predictions.shape[0], 1))
    for j in range(predictions.shape[0]):
        left_term[j, 0] = np.sum(beta[j, :] * inv_var[j, :])
        beta_sums[j, 0] = np.sum(beta[j, :])
        right_term[j, 0] = inv_prior_var * (1 - beta_sums[j])
    rbcm_inv_var = left_term + right_term

    # Compute Eq. 21
    # TODO: there is probably an np.einsum implementation possible,
    # maybe we don't need cython
    preds = np.zeros((predictions.shape[0], predictions.shape[1]))
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            summation = 0
            for k in range(predictions.shape[2]):
                summation += beta[i, k] * inv_var[i, k] * predictions[i, j, k]
            preds[i, j] = summation

    rbcm_var = 1 / rbcm_inv_var
    for j in range(predictions.shape[1]):
        preds[:, j] = rbcm_var[:, 0] * preds[:, j]
    return preds, rbcm_var