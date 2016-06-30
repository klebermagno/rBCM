# rBCM - robust Bayesian Committee Machine

A python package for fitting a robust Bayesian Committe Machine to data and
then predicting with it. This is much more highly scalable than Gaussian
Process Regression while still providing a good approximation to a full GPR
model. The package is built on top of and found much inspiration from the
GaussianProcessRegressor class found in scikit-learn, the interface is nearly
identical to that of the GPR class.

See the following paper for more information on the statistics of this
modeling approach. The description of the rBCM within was the foundation for
this implementation.

* jmlr.org/proceedings/papers/v37/deisenroth15.pdf

In concept, a robust Bayesian Committee Machine could be run massively in
parallel in a distributed environment on huge datasets. Though this
implementation is parallelized only by using the python multiprocessing
library.

Currently, an rBCM is passed a single kernel and that is used as the kernel
for all the experts. This must be one of the kernel subclasses from scikit-
learn.gaussian_process.kernels.

## Description

By fitting many smaller experts and then weighting them individually by their
uncertainty to a good-enough approximation of a single full model, the high
computational complexity of GPR can be managed.

This frees the user to consciously trade-off between accuracy and reduced
computational time for their application's needs in a way that many other
statistical techniques do not support. This trade-off is made by lowering or
raising the number data points assigned to each expert. The result is a
correspondingly increased or decreased total number of experts. This enables
more parallelism (each expert's fitting and prediction step is entirely
independent of all the others), but each expert is going to be making slightly
worse predictions.

This package considers a 'good' prediction as being close to the prediction of
a full GPR model fit to the entire dataset. The goal is to mirror the results
of Gaussian Process Regression with better computational performance.

However, it's called robust for a reason. Even with a very small number of
points per expert the predictions generated are quite reasonable approximations
to the full GPR model which would take orders of magnitude longer to fit.

An rBCM has two areas in which there is significant choice in implementation.
The first is in deciding how to partition the data among the experts. The
second is in deciding how to weight the predictions of the experts.

## Partitioning

This package currently supports two ways to partition the data set among
experts. Which one is used is passed as a boolean argument at instantiation
time of the rBCM object with the `locality` parameter. If False, the random
approach is taken. If True, the clustering approach is used.

#### Random

The first is simple a random partitioning of the data into equally-sized
chunks. This may be desirable for two reasons (which I have copied from the
white paper given above):

1. The experts may not always benefit from clustering, though it won't
   likely make the predictions worse it may not improve them either. This is
   demonstrated and discussed in the white paper above.
2. The computational complexity of clustering is non-trivial and may need to be
   avoided depending on the user's needs.

#### Clustered

This approach employs the Birch clustering algorithm implemented in scikit-
learn to cluster the dataset into as many clusters as there will be experts in
the rBCM. The virtue of this is that each expert will have, in the region at
which it is a local expert (now the 'expert' nomenclature makes more sense)
all the possible data for that location. It will not have some subsampling of
the data in that region given by random chance, it will have all of it. 

This should lead to the weighting procedure finding one expert at each
prediction location which has very high certainty, while all the other models
should have very low certainty. This sometimes has been observed to induce
predictions closer to that of the full GPR, but not always.


## Weighting

The core idea that enables the rBCM is that a Gaussian Process Regression
model can provide a measure of uncertainty for its predictions, not just the
value of the prediction itself. Not all statistical models do this, and as
such this approach of divying up data into experts and then merging them is
not always applicable when attempting to use other models as the underlying
experts.

The exact weighting algorithm is given in the paper above on page 5 and
involves computing a weight `beta_k` for each prediction location for each
expert k. This beta is taken to be the differential entropy of the posterior
from the prior. In pseudocode, for a single prediction location x and a single
expert k the value of beta_k is given as:

```python
beta_k = (0.5) * (log(prior_variance) - log(posterior_variance(expert_k, x)))
```

This term is then used as a scaling factor when computing the merged
prediction from all the expert's predictions as well as when computing the
overall variance of the rBCM's prediction.


## Installation

To install rBCM, run `sudo python setup.py install` inside the rBCM top-level directory.
To clean up all the build files generated, call `python setup.py clean`.
To run the test suite, navigate to the tests directory cand call `py.test`.

This package requires:
* Numpy
* Cython
* scikit-learn
