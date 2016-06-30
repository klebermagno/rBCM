"""This module contains the weighting algorithms used to find a single
predicted value from the weighted predictions of many experts.
"""
import time 

# cython related imports
import cython
cimport cython
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef differential_entropy_weighting(np.ndarray[np.float64_t, ndim=3] predictions,
                                    np.ndarray[np.float64_t, ndim=2] sigma,
                                    double prior_std):
    """Take the predictions from all the experts and weight them by
    how uncertain they are about their own predictions to create a
    single returned value that approximates a full gaussian process
    regression model's output.

    The procedure used is described mathematically on page 5 in:
        jmlr.org/proceedings/papers/v37/deisenroth15.pdf

    The uncertainty measure used is the differential entropy between the prior
    and the posterior at the location of the prediction. A model who has a
    posterior with significantly lower entropy at the location of the
    prediction than the prior's entropy probably carries a lot of good
    predictive information and we upweight that. A model we want to downweight
    would not have decreased the entropy very much, because it was not assigned
    relevant data that would make it more certain about its prediction.

    Parameters:
        predictions:.numpy array (n_locations, y_num_columns, num_experts)

        sigma:.......numpy array (n_locations, num_experts)
                     The uncertainty of each expert at each location

        prior_std:...python float indicating the standard deviation of the prior
    """
    # Just declaring everything up front c style for the cython compiler
    cdef double t_main = time.time()
    cdef double summation
    cdef int i, j, k
    cdef np.ndarray[np.float64_t, ndim=2] log_var, inv_var, beta, beta_sums, left_term, right_term
    cdef np.ndarray[np.float64_t, ndim=2] preds
    cdef double prior_var, log_prior_var, inv_prior_var

    # Precompute a bunch of terms used later
    var = np.power(sigma, 2)
    log_var = np.log(var)
    inv_var = 1 / var

    prior_var = np.power(prior_std, 2)
    log_prior_var = np.log(prior_var)
    inv_prior_var = 1 / prior_var

    num_preds = sigma.shape[0]
    num_experts = sigma.shape[1]
    pred_output_dim = predictions.shape[1]

    # Compute beta weights, page 5 right hand column
    beta = np.zeros((num_preds, num_experts))
    for j in range(num_experts):
        beta[:, j] = (0.5) * (log_prior_var - log_var[:, j])

    # Compute Eq. 22, page 5 left hand column
    beta_sums = np.zeros((num_preds, 1))
    left_term = np.zeros((num_preds, 1))
    right_term = np.zeros((num_preds, 1))
    for j in range(num_preds):
        left_term[j, 0] = np.sum(beta[j, :] * inv_var[j, :])
        beta_sums[j, 0] = np.sum(beta[j, :])
        right_term[j, 0] = inv_prior_var * (1 - beta_sums[j])
    rbcm_inv_var = left_term + right_term

    # Compute Eq. 21, page 5 left hand column
    # TODO: there is probably an np.einsum implementation possible
    rbcm_var = 1 / rbcm_inv_var
    preds = np.zeros((num_preds, pred_output_dim))
    for i in range(num_preds):
        for j in range(pred_output_dim):
            summation = 0
            for k in range(num_experts):
                summation += beta[i, k] * inv_var[i, k] * predictions[i, j, k]
            preds[i, j] = summation
    for j in range(pred_output_dim):
        preds[:, j] = rbcm_var[:, 0] * preds[:, j]

    time_passed = time.time() - t_main
    print('Weighting predictions took ' + str(int(time_passed)) + '.' + str(int(((time_passed % 1) * 100))) + ' seconds')
    return preds, rbcm_var
