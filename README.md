# rBCM - robust Bayesian Committee Machine

A python package for fitting a robust Bayesian Committe Machine (rBCM) to data and
then predicting with it. This is much more highly scalable than Gaussian
Process regression (GPR) while still providing a good approximation to a full GPR
model.

The package is built on top of and found much inspiration from the
gaussian_process package found in scikit-learn.

See the paper below for more information on the statistics of this modeling
approach. The description of the rBCM within was the foundation for this
implementation. There is also interesting comparison of rBCM with the Bayesian
Committee Machine and General Product of Experts models; how they are closely
related and how their performance differs statistically and computationally.

* [Distributed Gaussian Processes](jmlr.org/proceedings/papers/v37/deisenroth15.pdf)

In concept, a robust Bayesian Committee Machine could be run in parallel in a
distributed environment on huge datasets. Though this implementation is
parallelized only by using the python multiprocessing library.

## Installation

To install rBCM, run `sudo python setup.py install -i` inside the rBCM top-
level directory. To clean up all the build files generated, call `python
setup.py clean`.

This package requires:
* Numpy
* Cython
* scikit-learn

To run the test suite, navigate to the tests directory and call `nose2`.

The test suite builds plots in static html files as visual tests that are
placed in the `tests/visuals/` directory and requires the bokeh package for
this. If you don't use the test suite you don't need bokeh though.

## Why use an rBCM?

rBCM models offer the attractions of GPR modeling without as disabling of
computational issues. They are useful as a non-parametric Bayesian approach to
supervised learning. You can leverage an informative prior and not worry about
fiddling with hyperparameters. Gaussian Process regression permits exact
Bayesian inference and is known to have good performance (in statistical
terms, not computational).

By fitting many smaller experts and then weighting them individually by their
uncertainty to a good-enough approximation of a single full model, the high
computational complexity of GPR can be purposefully managed.

This frees the user to consciously trade-off between accuracy and reduced
computational time for their application's needs in a way that many other
statistical techniques do not support. This trade-off is made by lowering or
raising the number data points assigned to each expert with the
`points_per_expert` parameter. The result is a correspondingly increased or
decreased total number of experts. This enables more parallelism (each
expert's fitting and prediction step is entirely independent of all the
others), but each expert is going to be making slightly worse predictions.

However, it's called robust for a reason. Even with a surprisingly small
number of points per expert the predictions generated are quite reasonable
approximations to the full GPR model which might take an order of magnitude
longer to fit.

The choice of points per expert should therefore be dictated by your
performance needs. Set it as high as is tolerable for your application's speed
requirements. But make sure that it is low enough that the data set gets split
up at least as many times as you have cpu cores or you might start seeing a
performance falloff.

This package considers a 'good' prediction as being close to the prediction of
a full GPR model fit to the entire dataset. The goal is to mirror the results
of GPR with better computational performance.

## Usage

The following are different considerations you should be aware of when using
this package. Though you may easily use the package without looking at any of
these things, here is an example:

```python
import numpy as np
from rBCM.rbcm import RobustBayesianCommitteeMachineRegressor as RBCM
machine = RBCM()

def func(X):
    return np.sin(X) + X**2

X = np.arange(10000, dtype=np.float64).reshape(-1, 1) / 100
y = func(X)
machine.fit(X, y)
predictions = machine.predict(X)
```

### Kernel

Currently, an rBCM is passed a single kernel and that is used as the kernel
for all the experts. This must be one of the kernel subclasses from the
`scikit-learn.gaussian_process.kernels` module. This is a choice of prior and
you may pass any you like, but know that each expert's parameters get
optimized independently of one another to only the data partitioned to that
expert.

### Batching

In the prediction step, you can pass a `batch_size` parameter to instruct the
experts to predict at only `batch_size` data locations in parallel at once.
This can be used to limit maximum system memory consumption. Prediction with
GPR models is known to be a memory hog. Though the default of `batch_size=-1`
sets there to be no batching by default and you may not need it depending on
how many predictions you request at once.

### Partitioning

This package currently supports two ways to partition the data set among
experts. Which one is used is passed as a boolean argument at instantiation
time of the rBCM object with the `locality` parameter. If False, the random
approach is taken. If True, the clustering approach is used.

#### Random

The first is simply a random partitioning of the data into as many equally-
sized chunks as there will be experts. This may be desirable for two reasons
(which I have ctaken from the paper above):

1. The experts may not always benefit from clustering, though it won't
   likely make the predictions worse it may not improve them either.
2. The computational complexity of clustering is non-trivial and may need to be
   avoided depending on the user's needs.

#### Clustered

This approach employs the Birch clustering algorithm implemented in scikit-
learn to cluster the dataset into as many clusters as there will be experts in
the rBCM. The virtue of this is that each expert will have, in the region in
which it is a local expert (now the 'expert' nomenclature makes more sense),
all the possible data for that location. It will not have some subsampling of
the data in that region given by random chance, it will have all of it.

This should lead to the weighting procedure finding one expert at each
prediction location which has very high certainty, while all the other models
should have very low certainty. This sometimes has been observed to induce
predictions closer to that of the full GPR, but not always.


### Weighting

The core idea that enables the rBCM is that a Gaussian Process regression
model can provide a measure of uncertainty for its predictions, not just the
value of the prediction itself. Not all statistical models do this, and as
such this approach of divying up data into experts and then merging them is
not always applicable when attempting to use other models as the underlying
experts.

The exact weighting formula is given in the paper at the top of this readme on
page 5 and involves first computing a weight `beta_k` for each prediction
location for each `expert_k`. It's not obvious what metric is best to use, but
the following paper makes some suggestions in section 2.3:

* [Generalized Product of Experts for Automatic and Principled Fusion of Gaussian Process Predictions](http://arxiv.org/pdf/1410.7827v2.pdf)

This package currently only supports one choice of beta metric for now, though
adding more is high on the todo list. The beta is taken to be half the
difference in the log of the variance of the posterior from the log of the
variance of the prior. This is also known as the difference in differential
entropy between the posterior and prior.

This has been chosen because it indicates how well the GPR model generalizes
accurately at the prediction location from its training. If the variance of
the posterior is meaningfully smaller than that of the prior, then it likely
has relevant data and we upweight that prediction.

In pseudocode, for a single prediction location `x` and a single `expert_k` the value
of `beta_k` is given as:

```python
beta_k = (0.5) * (log(prior_variance) - log(posterior_variance(expert_k, x)))
```
Note that the `prior_variance` is a single constant value, while
`posterior_variance(expert_k, x)` depends on the specific expert and location.
This term is then used as a scaling factor when computing the merged
prediction from all the expert's predictions as well as when computing the
variance of the overall rBCM's predictions.


## Relevant References
    
* [Bayesian Committee Machine background information](http://www.dbs.ifi.lmu.de/~tresp/papers/bcm6.pdf)
* [Other alternative ways to scale gaussian process regression](http://www.dbs.ifi.lmu.de/~tresp/papers/nips02_approxgp.pdf)
