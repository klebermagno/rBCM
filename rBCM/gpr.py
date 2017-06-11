import numpy as np

from sklearn.gaussian_process.gpr import GaussianProcessRegressor as GPR


def fit(kernel, sample_indices, X, y, n_restarts_optimizer, normalize_y):
    """Fits a Gaussian Process Regression model on a subset of X and y using
    the provided covariance kernel and subset indices. This is used as a single
    worker payload in the parallel fitting process of the rBCM.

    TODO: take the sample_indices argument out of this function and keep it
          in the logic of the rBCM class alone. Just pass the X and y we'll
          actually use. For now keep it to avoid too many changes during the
          refactor, however.

    Args:
        kernel : sklearn kernel object
            The kernel specifying the covariance function of the Guassian
            Process.

        sample_indices : list of integers
            The indices of the subset of X and y to fit

        X : np.ndarray
            The locations of the points.
            Must match y in length.

        y : np.ndarray
            The values of the points at the X locations.
            Must match X in length.

        n_restarts_optimizer : non-negative integer
            The number of restarts to permit in the GPR. Look to scikit-learn's
            GPR implementation for more detail as it is passed through.

        normalize_y : boolean
            Whether to normalize the scale of y to improve fitting quality.
            See scikit-learn's GPR implementation for more detail.
    """
    gpr = GPR(kernel, n_restarts_optimizer=n_restarts_optimizer,
              copy_X_train=False, normalize_y=normalize_y)
    gpr.fit(X[sample_indices, :], y[sample_indices, :])
    return gpr


def predict(expert, X, y_num_columns):
    """Predicts using a Gaussian Process Regression model called expert to
    points given by X. This is used as a single worker payload in the parallel
    prediction process of the rBCM.

    Args:
        X : np.ndarray
            The locations of the points to predict at.
            Must match y in length.

        y_num_columns : positive integer
            The number of columns in the y data that was used during the
            fitting of the expert model.
    """
    predictions = np.zeros((X.shape[0], y_num_columns))
    sigma = np.zeros((X.shape[0], 1))
    predictions, sigma = expert.predict(X, return_std=True)
    return predictions, sigma
