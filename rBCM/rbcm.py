"""Robust Bayesian Committee Machine Regression"""

import multiprocessing as mp
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
from sklearn.gaussian_process.gpr import GaussianProcessRegressor as GPR
from sklearn.utils.validation import check_X_y, check_array

from .weighting import differential_entropy_weighting


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

        # Degenerate to full GPR for data with small n.
        # Change this before fitting to configure.
        self.min_points_for_committee = 2048

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
        # Enforce our type and shape requirements on X and y
        X = self._validate_input(X)
        y = self._validate_input(y)

        # Make sure they are consistent and sensical together, i.e. matching
        # sizes, consistent data types, etc.
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

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

        # We normalize all the data as one, not individually, and then save X
        # and y on the model object
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

        # we can't multiprocess without more than one expert anyways.
        if num_experts == 1:
            in_parallel = False

        # Bypass all the slow stuff if we are just making a point prediction.
        # Forking processes for such small amounts of work may not be worth it,
        # this choice of condition when to do this is pretty arbitrary though.
        if X.shape[0] == 1:
            predictions, sigma = self._simple_predict(X)

        # If not making point prediction we do all the large scale work though
        else:
            predictions, sigma = self._batchable_forkable_predict(
                    X, batch_size, in_parallel)

        if num_experts != 1:
            preds, var = differential_entropy_weighting(predictions, sigma,
                                                        self.prior_std)
        else:
            preds = predictions
            var = np.power(sigma, 2)

        # This changing of dimensions may be wrong? preds was coming out as
        # (100,) shape sometimes, but we want (100, 1).
        if len(preds.shape) == 1:
            preds = preds[:, np.newaxis]
        if len(var.shape) == 1:
            var = var[:, np.newaxis]

        if return_var:
            return predictions, var
        else:
            return preds

    def _batchable_forkable_predict(self, X, batch_size, in_parallel):
        """
        Do the prediction possibly in parallel and with batched points.
        """
        num_experts = len(self.experts)

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

        return predictions, sigma

    def _simple_predict(self, X):
        """
        Do the prediction without forking processes at all. Which may be useful
        for low volume predictions like point predictions that can be faster
        without bothering to multiprocess.
        """
        num_experts = len(self.experts)

        predictions = np.empty((1, self.y.shape[1], num_experts))
        sigma = np.empty((1, num_experts))
        for i in range(num_experts):
            predictions[:, :, i], sigma[:, i] = self.experts[i].predict(X)
        return predictions, sigma

    def _validate_input(self, array):
        """
        Validates the input for X or y at the fitting or prediction stages.
        """
        if type(array) is not np.ndarray:
            raise ValueError("Cannot provide X, y inputs \
that are not numpy.ndarrays objects.")

        return np.atleast_2d(array)

    def _generate_partitioned_indices(self):
        """
        Return a list of lists each containing a partition of the indices of
        the data to be fit.
        """
        # If not degenerate, but passed no ppe
        _DEFAULT_POINTS_PER_EXPERT = int(
                float(self.num_samples) / mp.cpu_count())

        if self.points_per_expert is None:
            if self.num_samples >= self.min_points_for_committee:
                self.points_per_expert = _DEFAULT_POINTS_PER_EXPERT
            else:
                self.points_per_expert = self.num_samples

        sample_sets = []
        indices = np.arange(self.num_samples)

        if self.locality:
            num_clusters = int(
                    float(self.num_samples) / self.points_per_expert)
            birch = Birch(n_clusters=num_clusters, threshold=0.2)
            labels = birch.fit_predict(self.X)
            unique_labels = np.unique(labels)

            # Fill each inner list i with indices matching its label i
            for label in unique_labels:
                sample_sets.append([i for i in indices if labels[i] == label])
        else:
            np.random.shuffle(indices)

            cap = self.num_samples
            ppe = self.points_per_expert

            for i in range(0, cap, ppe):
                if (i + ppe) >= cap:
                    ppe = cap - i
                sample_sets.append(indices[i:i + ppe])
        return sample_sets


def _worker_fit(kernel, sample_indices, X, y,
                n_restarts_optimizer, normalize_y):
    """This contains the parallel workload used in the fitting of the rbcm"""
    gpr = GPR(kernel, n_restarts_optimizer=n_restarts_optimizer,
              copy_X_train=False, normalize_y=normalize_y)
    gpr.fit(X[sample_indices, :], y[sample_indices, :])
    return gpr


def _worker_predict(expert, X, y_num_columns):
    """Parallel workload for predicting a single expert"""
    predictions = np.zeros((X.shape[0], y_num_columns))
    sigma = np.zeros((X.shape[0], 1))
    predictions, sigma = expert.predict(X, return_std=True)
    return predictions, sigma


# These simply unpack arguments for the fitting and prediction worker function.
_worker_fit_wrapper = lambda args: _worker_fit(*args)
_worker_predict_wrapper = lambda args: _worker_predict(*args)
