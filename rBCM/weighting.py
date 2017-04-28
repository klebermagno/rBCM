"Weighting and combination of sets of predictions into a single prediction."

import numpy as np


def differential_entropy_weighting(predictions, sigma, prior_std):
    """Weight the predictions of experts and reduce to a single prediction.

    This formula can be described as:
        The differential entropy between the prior predictive distribution and
        the posterior predictive distribution.

    See 'jmlr.org/proceedings/papers/v37/deisenroth15.pdf' page 5 for a better
    description of this weighting and the exact formulas.

    Args:
        predictions : array, shape = (n_locations, y_num_columns, num_experts)
            The values predicted by each expert

        sigma : array, shape = (n_locations, num_experts)
            The uncertainty of each expert at each location

        prior_std : float
            The standard deviation of the prior used to fit the GPRs

    Returns:
        preds : array, shape = (n_locations, y_num_columns)
            Mean of predictive distribution at query points

        rbcm_var : array, shape = (n_locations, )
            Variance of predictive distribution at query points
    """
    # We sometimes can be given zeros here and cannot deal with it, so set any
    # zeros to a very small number.
    sigma[sigma == 0] = 1E-9
    var = np.power(sigma, 2)
    log_var = np.log(var)
    prior_var = np.power(prior_std, 2)
    log_prior_var = np.repeat(np.log(prior_var), sigma.shape[0])

    # Compute beta weights, page 5 right hand column
    beta = 0.5 * (log_prior_var[:, np.newaxis] - log_var[:, :])

    # Combine the experts according to their beta weight
    # print(predictions[0, :, :])
    preds_old, var_old = _combine_old(predictions, var, beta, prior_var)
    # print(preds_old[0, :])
    # preds, var = _combine(predictions, var, beta, prior_var)
    # np.allclose(preds_old, preds)
    # np.allclose(var_old, var)
    return preds_old, var_old


def _combine(predictions, var, beta, prior_var):
    """Calculate a single prediction from many with the given beta weights.

    This should be able to accept any general measure of uncertainty, beta.

    Args:
        predictions : array-like, shape = (n_locations, n_features, n_experts)
            Values predicted by some sklearn predictor that offers var as well

        var : array-like, shape = (n_locations, n_experts)
            Variances corresponding to the predictions

    Returns:
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
    rbcm_var = 1 / rbcm_inv_var
    preds = np.einsum("ik, ik, ijk->ij", beta, inv_var, predictions)
    preds = rbcm_var[:, np.newaxis] * preds[:, :]

    return preds, rbcm_var


def _combine_old(predictions, var, beta, prior_var):
    """Calculate a single prediction from many with the given beta weights.

    This should be able to accept any general measure of uncertainty, beta.

    This is the OLD version of this function which is based on naive loops, but
    is easier to understand logically and check for correctness. This (hopeful)
    correctness is used to verify correctness of the new version which utilizes
    numpy's einsum() function for performance.

    Args:
        predictions : array-like, shape = (n_locations, n_features, n_experts)
            Values predicted by some sklearn predictor that offers var as well

        var : array-like, shape = (n_locations, n_experts)
            Variances corresponding to the predictions

    Returns:
        predictions : array, shape = (n_locations, n_features)

        rbcm_var : array, shape = (n_locations)
    """
    inv_var = 1 / var
    inv_prior_var = 1 / prior_var

    # Compute Eq. 22
    beta_sums = np.zeros(predictions.shape[0])
    left_term = np.zeros(predictions.shape[0])
    right_term = np.zeros(predictions.shape[0])
    for j in range(predictions.shape[0]):
        left_term[j] = np.dot(beta[j, :], inv_var[j, :])
        beta_sums[j] = np.sum(beta[j, :])
        right_term[j] = inv_prior_var * (1 - beta_sums[j])
    rbcm_inv_var = left_term + right_term

    # Computer Eq. 21
    rbcm_var = 1 / rbcm_inv_var
    rbcm_var = rbcm_var
    preds = np.zeros((predictions.shape[0], predictions.shape[1]))
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            summation = 0
            for k in range(predictions.shape[2]):
                summation += beta[i, k] * inv_var[i, k] * predictions[i, j, k]
            preds[i, j] = summation

    for i in range(preds.shape[0]):
        preds[i, :] = rbcm_var[i] * preds[i, :]

    return preds, rbcm_var
