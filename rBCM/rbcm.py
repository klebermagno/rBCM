"""Robust Bayesian Committee Machine Regression"""

# Authors: Lucas Jere Kolstad <lucaskolstad@gmail.com>
#
# License: See LICENSE file

import multiprocessing as mp
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
from sklearn.gaussian_process.gpr import GaussianProcessRegressor as GPR
from sklearn.utils.validation import check_X_y, check_array

from weighting import differential_entropy_weighting


class RobustBayesianCommitteeMachineRegressor(BaseEstimator, RegressorMixin):
    """Robust Bayesian Committee Machine Regression (rBCM).

    See 'jmlr.org/proceedings/papers/v37/deisenroth15.pdf' for the whitepaper
    describing the statistical behavior which this implementation provides.

    Args:
        kernel : sklearn kernel object
            The kernel specifying the covariance function of each GP expert.
            If None is passed, "1.0 * RBF(1.0) + WhiteKernel(1e-10)" is used
            as the default kernel. Note that the WhiteKernel term means this
            is different from the sklearn.gaussian_process.gpr default, though
            the alpha=1e-10 default from gpr is set to 0 implicitly here to
            compensate.

        points_per_expert : integer (default: None)
            The maximum number of points to assign to each expert. Smaller
            causes there to be more experts (more parallelism), but each has
            fewer data points to train on and accuracy may degrade. The default
            None permits the RBCM implementation to decide however it wants.

        locality : boolean (default : False)
            False to randomly assign data to experts. True to use Birch
            clustering to assign data in local groups to experts

        max_points : integer (default : None)
            A cap on the total data the RBCM will look at when fit, best used
            for development to test on a subset of a large dataset quickly

        n_restarts_optimizer : non-negative integer (default : 0)
            The number of restarts each GPR gets on it's optimization
            procedure, the restarts pull parameters as described in the module
            sklearn.gaussian_process.gpr. This is simply passed on to that
            class.

        standardize_y : boolean (default : False)
            Whether to de-mean and scale the target variables. Simply passed on
            to the gpr class.

        standardize_X : boolean (default : True)
            Whether to de-mean and scale the predictor variables using the
            sklearn StandardScaler. The scaling is automatically handled at
            prediction time as well for the prediction locations. This can
            improve both predictive performance and computational time in some
            cases.
    """
    def __init__(self, kernel=None, points_per_expert=None, locality=False,
                 max_points=None, n_restarts_optimizer=0, normalize_y=False,
                 standardize_X=False):
        self.locality = locality
        self.kernel = kernel
        self.points_per_expert = points_per_expert
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.standardize_X = standardize_X

    def fit(self, X, y):
        """Fits X to y with a robust Bayesian Committee Machine model.

        Note that the rBCM retains a copy of the training data implicitly as
        each GPR expert retains a reference to their partition for prediction.

        Args:
            X : array, shape = (n_samples, n_features)

            y : array, shape = (n_samples, n_output_dims)

        Returns:
            self : returns an instance of self.
        """
        if self.n_restarts_optimizer < 0:
            self.n_restarts_optimizer = 0

        if self.kernel is None:
            # Could assign self.kernel explicitly to the default kernel
            # used by gpr class, but instead lets just hard code the prior's
            # std that we'll need and let the gpr class set the default kernel
            self.prior_std = 1
        else:
            self.prior_std = self.kernel(self.kernel.diag(np.arange(1)))[0]
        self.num_samples = X.shape[0]

        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        # We normalize all the data as one, not individually
        if self.standardize_X:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(X)
        else:
            self.X = X
        self.y = y

        # Partition the indices of the data into sample_sets
        self.sample_sets = self._generate_partitioned_indices()

        # Generate iterable list of arguments to map out to procs
        args_iter = []
        for samples in self.sample_sets:
            args_iter.append([self.kernel, samples, self.X, self.y,
                              self.n_restarts_optimizer, self.normalize_y])

        # Actually do the fitting on each partition now
        self.experts = []
        if len(args_iter) != 1:
            pool = mp.Pool()
            experts = pool.imap(_worker_fit_wrapper, args_iter)
            for i in range(len(self.sample_sets)):
                self.experts.append(experts.next())
            pool.close()
        else:
            self.experts.append(_worker_fit_wrapper(args_iter[0]))
        return self

    def predict(self, X, return_var=False, batch_size=None, in_parallel=True):
        """Predict using the robust Bayesian Committee Machine model

        Args:
            X : array, shape = (n_locations, n_features)
                Query points where the rBCM is evaluated

            batch_size : integer
                When given a large set of points at which to create
                predictions, this parameter can limit processing to a given
                batch size of the prediction points at once.

                The default None means no batching.

                This is useful for creating fuzzy upper memory bounds in cases
                where you ask the experts to, in parallel, each compute a
                prediction at a great many points.

            in_parallel: boolean
                Whether to use a parallel implementation or not to predict at
                each location in X. If only predicting a single point or if
                predicting with only one expert, this arg is ignored. For small
                X, consider setting false.

        Returns:
            y_mean : array, shape = (n_locations, n_features)
                Mean of the predictive distribution at the query points.

            y_var : array, shape = (n_locations, 1)
                Variance of the predictive distribution at the query points.
                Only returned if return_var is True.
        """
        X = check_array(X)
        if self.standardize_X:
            X = self.scaler.transform(X)
        num_experts = len(self.experts)

        if num_experts == 1:
            in_parallel = False

        # Bypass all the slow stuff if we are just making a point prediction
        if X.shape[0] == 1:
            predictions = np.empty((1, self.y.shape[1], num_experts))
            sigma = np.empty((1, num_experts))
            for i in range(num_experts):
                predictions[:, :, i], sigma[:, i] = self.experts[i].predict(X)

        # If not making point prediction we do all the large scale work though
        else:
            sigma = np.zeros((X.shape[0], num_experts))
            predictions = np.zeros((X.shape[0], self.y.shape[1], num_experts))
            args_iter = []

            if batch_size is None:
                batch_size = X.shape[0] * 2

            if in_parallel:
                pool = mp.Pool()

            # Batch the data and predict
            for i in range(0, X.shape[0], batch_size):
                batch_range = range(i, min(i + batch_size, X.shape[0]))
                batch = X[batch_range, :]

                # Generate iterable list of arguments to map out to procs
                # TODO: Could probably be passing a lot less around between
                # processes
                if len(args_iter) == 0:
                    for i in range(num_experts):
                        args_iter.append([self.experts[i],
                                          batch,
                                          self.y.shape[1]])

                # We only change the batch arg each loop
                if len(args_iter) != 0:
                    for j in range(num_experts):
                        args_iter[j][1] = batch

                # Using a simple chunk size policy for now
                chunkersize = int(np.rint(float(num_experts) / mp.cpu_count()))
                if chunkersize <= 0:
                    chunkersize = 1

                if in_parallel:
                    results = pool.imap(_worker_predict_wrapper,
                                        args_iter,
                                        chunksize=chunkersize)
                elif num_experts == 1:
                    predictions, sigma = _worker_predict_wrapper(args_iter[0])
                else:
                    results = map(_worker_predict_wrapper, args_iter)

                # Fill the preallocated arrays with the results as they come in
                for j in range(num_experts):
                    if in_parallel:
                        predictions[batch_range, :, j], sigma[batch_range, j] = results.next()
                    elif num_experts != 1:
                        predictions[batch_range, :, j], sigma[batch_range, j] = results[j]

            if in_parallel:
                pool.close()

        if num_experts != 1:
            preds, var = differential_entropy_weighting(predictions, sigma, self.prior_std)
        else:
            preds = predictions.ravel()
            var = np.power(sigma, 2).ravel()

        if return_var:
            return preds, var
        else:
            return preds

    def _generate_partitioned_indices(self):
        """Return a list of lists each containing a partition of the indices
        of the data to be fit.
        """
        # Degenerate to full GPR for data with small n
        _MIN_POINTS_FOR_RBCM = 2048
        # If not degenerate, but passed no ppe
        _DEFAULT_POINTS_PER_EXPERT = int(float(self.num_samples) / mp.cpu_count())

        if self.points_per_expert is None:
            if self.num_samples >= _MIN_POINTS_FOR_RBCM:
                self.points_per_expert = _DEFAULT_POINTS_PER_EXPERT
            else:
                self.points_per_expert = self.num_samples

        sample_sets = []
        indices = np.arange(self.num_samples)

        if self.locality:
            num_clusters = int(float(self.num_samples) / self.points_per_expert)
            birch = Birch(n_clusters=num_clusters, threshold=0.2)
            labels = birch.fit_predict(self.X)
            unique_labels = np.unique(labels)

            # Fill each inner list i with indices matching its label i
            for label in unique_labels:
                sample_sets.append([i for i in indices if labels[i] == label])
        else:
            np.random.shuffle(indices)

            # Renaming just for shorter names
            cap = self.num_samples
            exp = self.points_per_expert

            for i in range(0, cap, exp):
                if (i + exp) >= cap:
                    exp = cap - i
                sample_sets.append(indices[i:i + exp])
        return sample_sets


def _worker_fit(kernel, sample_indices, X, y,
                n_restarts_optimizer, normalize_y):
    """This contains the parallel workload used in the fitting of the rbcm"""
    gpr = GPR(kernel, n_restarts_optimizer=n_restarts_optimizer,
              copy_X_train=False, normalize_y=normalize_y)
    gpr.fit(X[sample_indices, :], y[sample_indices, :])
    return gpr


def _worker_fit_wrapper(args):
    """Just used to unpack arguments for parallel call to worker_fit"""
    return _worker_fit(*args)


def _worker_predict(expert, X, y_num_columns):
    """Parallel workload for predicting a single expert"""
    predictions = np.zeros((X.shape[0], y_num_columns))
    sigma = np.zeros((X.shape[0], 1))
    predictions, sigma = expert.predict(X, return_std=True)
    return predictions, sigma


def _worker_predict_wrapper(args):
    """Just used to unpack arguments for parallel call to worker_predict"""
    return _worker_predict(*args)
