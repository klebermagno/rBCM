"""Weighting and combination of sets of predictions into a single prediction."""

# Authors: Lucas Jere Kolstad <lucaskolstad@gmail.com>
#
# License: see LICENSE file

import numpy as np


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

    prior_var = np.power(prior_std, 2)
    log_prior_var = np.log(prior_var)

    # Compute beta weights, page 5 right hand column
    beta = np.zeros((sigma.shape[0], sigma.shape[1]))
    for j in range(predictions.shape[1]):
        beta[:, j] = (0.5) * (log_prior_var - log_var[:, j])

    return _combine(predictions, var, beta, prior_var)


def _combine(predictions, var, beta, prior_var):
    """Calculate a single prediction from many with the given beta weights.

    This should be able to accept any general measure of uncertainty, beta.

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
    inv_var = 1 / var
    inv_prior_var = 1 / prior_var

    # Compute Eq. 22
    left_term = np.einsum("ij, ij->i", beta, inv_var)
    right_term = inv_prior_var * (1 - np.einsum("ij->i", beta))
    rbcm_inv_var = left_term + right_term

    # Compute Eq. 21
    preds = np.einsum("ik, ik, ijk->ij", beta, inv_var, predictions)
    rbcm_var = 1 / rbcm_inv_var
    preds = rbcm_var[:, np.newaxis] * preds[:, :]
    return preds, rbcm_var
