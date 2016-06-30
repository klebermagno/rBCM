"""rBCM is a class used to fit and predict with a robust Bayesian Committee
Machine regression model on large data using many small, independent Gaussian
Process regression models.

See 'jmlr.org/proceedings/papers/v37/deisenroth15.pdf' for the whitepaper that
this rBCM implementation is based on.

An rBCM has an interface close to that of sklearn's GaussianProcessRegressor
class. What a RBCM does is handle behind-the-scenes the partitioning of your
data, training local expert models on those partitions, and predicting at
points you request by taking weighted predictions from individual models. It
is an approximation to a GPR and one which will work on much larger data sets.
"""
import time
import random
import multiprocessing as mp
import numpy as np

# sklearn stuff
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
from sklearn import gaussian_process as GPR

from weighting import differential_entropy_weighting


class rBCM:
    def __init__(self, kernel=None, points_per_expert=750, locality=False, max_points=100000):
        """Create a RBCM which will, when fitting, initialize all of it's
        future experts with the given arguments.

        Parameters:
            kernel:............The Kernel object from sklearn to be used
                               for each expert. Default None uses the
                               default for sklearn.gaussian_process.gpr

            points_per_expert:.The number of points to cap each expert,
                               smaller causes there to be more experts
                               (more parallelism), but each has less
                               information to accurately regress on

            locality:..........boolean - False to randomly assign data
                               to experts; True to use birch clustering
                               to assign data in local groups to experts

            max_points:........an integer cap on the total data the
                               RBCM will look at when fit, best used
                               for development to test on a subset
                               of a large dataset quickly
        """
        self.locality = locality
        self.max_points = max_points
        if kernel is None:
            kernel = C(1.0, constant_value_bounds="fixed") \
                * RBF(1.0, length_scale_bounds="fixed")
        self.kernel = kernel
        self.prior_std = kernel(kernel.diag(np.arange(1)))[0]
        self.points_per_expert = points_per_expert

    def fit(self, X, y, seed=None):
        """Fits X to y with a robust Bayesian Committee Machine model.

        Parameters:
            X:    numpy array (n_samples, n_features)
            y:    numpy array (n_samples, [n_output_dims])
            seed: optional integer used as the seed in randomly partitioning
                  the data into experts. If rBCM initialized with locality=True,
                  (i.e. we are clustering the data), then this is ignored.
        """
        print('Fitting a robust Bayesian Committee Machine to the partitions...')
        t = time.time()
        # Subsample down to max_points
        self.num_samples = X.shape[0]
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
        sample_sets = self._generate_partitioned_indices(seed)
        self.num_experts = len(sample_sets)

        # Generate iterable list of arguments to map out to procs
        args_iter = []
        for samples in sample_sets:
            args_iter.append((self.kernel, samples, self.X, self.y))

        # Actually do the parallel fitting on each partition now
        print('Fitting ' + str(self.num_experts) + ' experts...')
        pool = mp.Pool()
        experts = pool.imap(_worker_fit_wrapper, args_iter)

        # Transfer the experts from an IMapIterator object to a list
        self.experts = []
        t_sub = time.time()
        counter = 0
        for expert in experts:
            counter += 1
            time_passed = time.time() - t_sub
            print('Fit expert ' + str(counter) + ' of ' + str(self.num_experts) + ' after ' + str(int(time_passed)) + '.' + str(int(((time_passed % 1) * 100))) + ' seconds...')
            self.experts.append(expert)

        pool.close()
        time_passed = time.time() - t
        print('rBCM fitting took ' + str(int(time_passed)) + '.' + str(int(((time_passed % 1) * 100))) + ' seconds')

    def predict(self, X, batch_size=-1, parallel=True):
        """Returns the predicted values of y at X. Automatically handles
        scaling the X input with the same scaler used to scale the training
        data in fit(), ensuring consistent predictions. Does not return a
        sigma or cov estimate the way GPR can.

        Changing the options in this class can have widely varying performance
        impact. Batch_size chunks the number of rows in X into batches and ships
        those batches off one at a time to be processed in parallel, rather than
        shipping off the entirety of X. This is simply to conserve memory, but is
        not always needed and definitely slows so look at some perf metrics.

        Parameters:
            X:          numpy array - (n_locations, n_features)

            batch_size: integer indicating a batch_size to predict at once,
                        this limits memory consumption when predicting
                        on large X. The default -1 means no batching.

            parallel:   Whether to use a parallel implementation or not to predict
                        at each location in X. If only predicting a single point,
                        this arg is ignored. For small X, consider setting false.

        Returns:
            Predictions:  Numpy array (n_locations, n_features)
            rbcm_var:     Numpy array (n_locations, 1)
        """
        # Bypass all the slow stuff below if we are just making a point prediction
        if X.shape[0] == 1:
            scaled_X = self.scaler.transform(X)
            for i in range(len(self.experts)):
                predictions, sigma = self.experts[i].predict(scaled_X)

        # If not making point prediction we do all the large scale work though
        else:
            print('Prepping to calculate ' + str(self.num_experts) + ' expert\'s predictions...')
            t_main = time.time()

            X = self.scaler.transform(X)

            pool = mp.Pool()

            num_experts = len(self.experts)
            sigma = np.zeros((X.shape[0], num_experts), dtype='float64')
            predictions = np.zeros((X.shape[0], self.y.shape[1], num_experts), dtype='float64')

            batch_counter = 1
            args_iter = []
            in_batches = True
            if batch_size == -1:
                in_batches = False
                batch_size = X.shape[0] * 2

            # Batch the data and predict
            for i in range(0, X.shape[0], batch_size):
                t_sub = time.time()
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
                if in_batches:
                    print("Starting prediction batch " + str(batch_counter) + ' of ~' + str(int(X.shape[0] / float(batch_size)) + 1) + '!')
                if parallel:
                    results = pool.imap(_worker_predict_wrapper, args_iter, chunksize=chunkersize)
                else:
                    results = map(_worker_predict_wrapper, args_iter)

                # Fill the preallocated arrays with the results as they come in
                for j in range(num_experts):
                    if parallel:
                        predictions[batch_range, :, j], sigma[batch_range, j] = results.next()
                    else:
                        predictions[batch_range, :, j], sigma[batch_range, j] = results[j]
                    time_passed = time.time() - t_sub
                    print('Got expert ' + str(j + 1) + ' of ' + str(num_experts) + ' back after ' + str(int(time_passed)) + '.' + str(int(((time_passed % 1) * 100))) + ' seconds')

                if in_batches:
                    time_passed = time.time() - t_sub
                    print('Batch ' + str(batch_counter) + ' of ~' + str(int(X.shape[0] / float(batch_size)) + 1) + ' took ' + str(int(time_passed)) + '.' + str(int(((time_passed % 1) * 100))) + ' seconds')
                batch_counter += 1

            pool.close()
            time_passed = time.time() - t_main
            print('rBCM model\'s unweighted prediction took ' + str(int(time_passed)) + '.' + str(int(((time_passed % 1) * 100))) + ' seconds')

        return differential_entropy_weighting(predictions, sigma, self.prior_std)

    def _generate_partitioned_indices(self, seed):
        """Return a list of lists of integers, each inner list contains at most
        points_per_expert integer values partitioned from [0, num_samples-1].

        Behavior is set by the following RBCM attributes which must be preset:
            self.num_samples:       integer size of the indices to sample from

            self.points_per_expert: number of values to assign to each expert

            self.locality:          boolean - whether to randomly assign data
                                    to experts or localize with birch clustering
            self.seed               integer seed for random shuffling, or may be
                                    None in which case no seed is explicitly set.
        """
        print("Partitioning the data into a set for each expert...")
        sample_sets = []
        indices = np.arange(self.num_samples)
        if self.locality:
            num_clusters = int(float(self.num_samples) / self.points_per_expert)

            print('Clustering the data to provide locality for experts with Birch algorithm...')
            t = time.time()
            birch = Birch(n_clusters=num_clusters)
            labels = birch.fit_predict(self.X)

            unique_labels = np.unique(labels)

            # Fill each inner list i with indices matching its label i
            for label in unique_labels:
                sample_sets.append([i for i in indices if labels[i] == label])

            time_passed = time.time() - t
            print('Clustering took ' + str(int(time_passed)) + '.' + str(int(((time_passed % 1) * 100))) + ' seconds')
        else:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(indices)

            cap = indices.shape[0]
            for i in range(0, cap, self.points_per_expert):
                if (i + self.points_per_expert) >= cap:
                    self.points_per_expert = cap - i
                samples = indices[i:i + self.points_per_expert]
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
