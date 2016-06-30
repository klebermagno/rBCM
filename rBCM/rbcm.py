"""Robust Bayesian Committee Machine Regression"""

# Authors: Lucas Jere Kolstad <lucaskolstad@gmail.com>
#
# License: See LICENSE file

import random
import multiprocessing as mp
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
from sklearn import gaussian_process as GPR
from sklearn.utils.validation import check_X_y, check_array

from .weighting import differential_entropy_weighting


class RobustBayesianCommitteeMachineRegressor(BaseEstimator, RegressorMixin):
    """Robust Bayesian Committee Machine Regression (rBCM).

    See 'jmlr.org/proceedings/papers/v37/deisenroth15.pdf' for the whitepaper
    describing the rBCM which this implementation is based on.

    Parameters
    ----------
    kernel : kernel object
        The kernel specifying the covariance function of each GP expert.
        If None is passed, "1.0 * RBF(1.0) + WhiteKernel(1e-10)" is used
        as the default kernel. Note that the WhiteKernel term means this
        is different from the sklearn.gaussian_process.gpr default, though
        the alpha=1e-10 default from gpr is set to 0 implicitly here to
        compensate.

    points_per_expert: integer (default: 750)
        The maximum number of points to assign to each expert. Smaller
        causes there to be more experts (more parallelism), but each has
        fewer data points to train on and accuracy may degrade.

    locality: boolean (default: False)
        False to randomly assign data to experts. True to use Birch
        clustering to assign data in local groups to experts

    max_points: integer (default: None)
        A cap on the total data the RBCM will look at when fit,
        best used for development to test on a subset of a large dataset
        quickly

    Attributes
    ----------

    """
    def __init__(self, kernel=None, points_per_expert=750, locality=False, max_points=None):
        self.locality = locality
        self.max_points = max_points
        self.kernel = kernel
        self.prior_std = kernel(kernel.diag(np.arange(1)))[0]
        self.points_per_expert = points_per_expert

    def fit(self, X, y):
        """Fits X to y with a robust Bayesian Committee Machine model.

        Note that the rBCM retains a copy of the training data implicitly as
        each GPR expert retains a reference to their partition for prediction.

        Parameters:
        ----------
        X : array, shape = (n_samples, n_features)

        y : array, shape = (n_samples, n_output_dims)

        Returns:
        ----------
        self : returns an instance of self.
        """
        if self.kernel is None:
            self.kernel = C(1.0, constant_value_bounds="fixed") \
                * RBF(1.0, length_scale_bounds="fixed")

        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        # Subsample down to max_points if parameter was set
        self.num_samples = X.shape[0]
        if self.max_points is not None:
            if self.max_points < self.num_samples:
                samples = random.sample(range(X.shape[0]), self.max_points)
                self.X = X[samples, :]
                self.y = y[samples, :]
                self.num_samples = self.max_points
            else:
                self.X = X
                self.y = y

        # We scale all the data as one, and do no additional scaling in
        # each individual expert
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        y_mean = np.mean(y, axis=0)
        y = y - y_mean

        # Partition the indices of the data into sample_sets
        if self.seed is None:
            sample_sets = self._generate_partitioned_indices()
        else:
            sample_sets = self._generate_partitioned_indices(self.seed)
        self.num_experts = len(sample_sets)

        # Generate iterable list of arguments to map out to procs
        args_iter = []
        for samples in sample_sets:
            args_iter.append((self.kernel, samples, self.X, self.y))

        # Actually do the parallel fitting on each partition now
        pool = mp.Pool()
        experts = pool.imap(_worker_fit_wrapper, args_iter)

        # Transfer the experts from an IMapIterator object to a list
        self.experts = []
        counter = 0
        for expert in experts:
            counter += 1
            self.experts.append(expert)
        pool.close()
        return self

    def predict(self, X, return_var=False, batch_size=-1, in_parallel=True):
        """Predict using the robust Bayesian Committee Machine model

        Parameters:
        ----------
        X : array, shape = (n_locations, n_features)
            Query points where the rBCM is evaluated
        batch_size : integer
            indicating a batch_size to predict at once, this limits memory
            consumption when predicting on large X. The default -1 means no
            batching.

        in_parallel: boolean
            Whether to use a parallel implementation or not to predict at each
            location in X. If only predicting a single point, this arg is
            ignored. For small X, consider setting false.

        Returns:
        ----------
        y_mean : array, shape = (n_locations, n_features)
            Mean of the predictive distribution at the query points.

        y_var : array, shape = (n_locations, 1)
            Variance of the predictive distribution at the query points.
            Only returned if return_var is True.
        """
        X = check_array(X)

        # Bypass all the slow stuff if we are just making a point prediction
        if X.shape[0] == 1:
            scaled_X = self.scaler.transform(X)
            for i in range(len(self.experts)):
                predictions, sigma = self.experts[i].predict(scaled_X)

        # If not making point prediction we do all the large scale work though
        else:
            X = self.scaler.transform(X)

            pool = mp.Pool()

            num_experts = len(self.experts)
            sigma = np.zeros((X.shape[0], num_experts), dtype='float64')
            predictions = np.zeros((X.shape[0], self.y.shape[1], num_experts), dtype='float64')

            batch_counter = 1
            args_iter = []
            if batch_size == -1:
                batch_size = X.shape[0] * 2

            # Batch the data and predict
            for i in range(0, X.shape[0], batch_size):
                batch_range = range(i, min(i + batch_size, X.shape[0]))
                batch = X[batch_range, :]

                # Generate iterable list of arguments to map out to procs
                # TODO: Could probably be passing a lot less around between procs
                if len(args_iter) == 0:
                    for expert in self.experts:
                        args_iter.append([expert, batch, self.y.shape[1]])

                # We only change the batch arg each loop, leave experts and y.shape alone
                if len(args_iter) != 0:
                    for j in range(len(self.experts)):
                        args_iter[j][1] = batch

                # Using a simple chunk size policy for now
                chunkersize = int(np.rint(float(num_experts) / mp.cpu_count()))
                if chunkersize == 0:
                    chunkersize = 1

                # Map out the heavy computation
                if in_parallel:
                    results = pool.imap(_worker_predict_wrapper, args_iter, chunksize=chunkersize)
                else:
                    results = map(_worker_predict_wrapper, args_iter)

                # Fill the preallocated arrays with the results as they come in
                for j in range(num_experts):
                    if in_parallel:
                        predictions[batch_range, :, j], sigma[batch_range, j] = results.next()
                    else:
                        predictions[batch_range, :, j], sigma[batch_range, j] = results[j]

                batch_counter += 1
            pool.close()
        preds, var = differential_entropy_weighting(predictions, sigma, self.prior_std)
        if return_var:
            return preds, var
        else:
            return preds

    def _generate_partitioned_indices(self, seed):
        """Return a list of lists each containing a partition of the indices
        of the data to be fit.

        Each inner list contains at most self.points_per_expert indices
        partitioned from the set [0, num_samples-1].

        If self.locality was set True, clusters the data to be fit and
        partitions each expert their own cluster.
        """
        sample_sets = []
        indices = np.arange(self.num_samples)
        if self.locality:
            num_clusters = int(float(self.num_samples) / self.points_per_expert)

            birch = Birch(n_clusters=num_clusters)
            labels = birch.fit_predict(self.X)

            unique_labels = np.unique(labels)

            # Fill each inner list i with indices matching its label i
            for label in unique_labels:
                sample_sets.append([i for i in indices if labels[i] == label])
        else:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(indices)

            # Renaming just for convenience
            cap = indices.shape[0]
            exp = self.points_per_expert

            for i in range(0, cap, exp):
                if (i + exp) >= cap:
                    exp = cap - i
                samples = indices[i:i + exp]
                sample_sets.append(samples)
        return sample_sets


def _worker_fit(kernel, sample_indices, X, y):
    """This contains the parallel workload used in the fitting of the rbcm"""
    gpr = GPR(kernel, normalize_y=False, copy_X_train=False, n_restarts_optimizer=3)
    gpr.fit(X[sample_indices, :], y[sample_indices, :])
    return gpr


def _worker_fit_wrapper(args):
    """Just used to unpack arguments for parallel call to worker_fit"""
    return _worker_fit(*args)


def _worker_predict(expert, X, y_num_columns):
    """Parallel workload for predicting a single expert"""
    predictions = np.zeros((X.shape[0], y_num_columns), dtype=np.float64)
    sigma = np.zeros((X.shape[0], 1), dtype=np.float64)
    predictions, sigma = expert.predict(X)
    return predictions, sigma


def _worker_predict_wrapper(args):
    """Just used to unpack arguments for parallel call to worker_predict"""
    return _worker_predict(*args)
