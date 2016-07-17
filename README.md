# rBCM - robust Bayesian Committee Machine

A python package for fitting a robust Bayesian Committee Machine (rBCM) to data and
then predicting with it. This is much more highly scalable than Gaussian
Process regression (GPR) while still providing a good approximation to a full GPR
model. The rBCM contains the exact GPR model as a special case.

Other alternatives to scaling GPR have undesirable downsides (discussed in
[reference 4](#ref4)). They are typically either too inaccurate to a full GPR
or have worse scalability characteristics than the rBCM.

If you use this package on a multicore machine you should expect to see at
least a 10x speed up from a full GPR. As you increase the size of the data you
are working with, this only improves. There is significant parallel scaling in
an rBCM not present in a GPR and running on a machine with twice as many cores
can oftentimes double the speed.

After a certain size GPR is also simply intractable, so the rBCM can compute
predictions that mimic a GPR prediction on data sets for which it is
practically infeasible to compute a real GPR prediction.

The package is built on top of and found much inspiration from the
`sklearn.gaussian_process` package found in scikit-learn.

See the paper below for more information on the statistics of this modeling
approach. The description of the rBCM within was the foundation for this
implementation. There is also interesting comparison of rBCM with the Bayesian
Committee Machine and General Product of Experts models; how they are closely
related and how their performance differs statistically and computationally.

* [Distributed Gaussian Processes](http://www.jmlr.org/proceedings/papers/v37/deisenroth15.pdf)

## Installation

To install rBCM, run `sudo python setup.py install -i` inside the rBCM top-
level directory. To clean up all the build files generated, call `python
setup.py clean`.

This package requires:
* Numpy
* scikit-learn

To run the test suite, navigate to the tests directory and call `nose2`.

The test suite and benchmark runners build plots in static html files as
visual tests that are placed in the `tests/visuals/` or benchmarks/visuals/
directory respectively and requires the bokeh package for this. If you don't
use the test suite or benchmarks you don't need bokeh though.

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

The following graphs show how the rBCM can perform just as well as a GPR. The
rBCM had 4 experts each trained on 256 randomly selected points out of the
total 1024 data points. Then these fitted experts were used to predict onto
2000 evenly spaced points to generate the predictive mean line and 95%
confidence band. The data was standardized before fitting and prediction as
well. 

This was run on my laptop with 4 cores (matching the 4 experts) and the rBCM
took only ~10 seconds compared to the ~53 seconds of the full GPR from
sklearn. This was also permitting every expert in the rBCM 2 restarts on its
parameter optimization runs, as well as 2 restarts for the full GPR model
itself.

![GPR versus rBCM Graphs](docs/graphs/gprVSrbcm_4experts.png?raw=true "Optional Title")

## Usage

The following are different considerations to be aware of when using this
package. Though you may easily use the package without looking at any of these
things, here is an example:

```python
import numpy as np
from rBCM.rbcm import RobustBayesianCommitteeMachineRegressor as RBCM
machine = RBCM()

def f1(x):
    return np.abs(np.sin(x)) * 5

X = np.linspace(10, 20, 100, dtype=np.float64)
y = f1(X)
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

##### Random

The first is simply a random partitioning of the data into as many equally-
sized chunks as there will be experts. This may be desirable for two reasons
(which I have taken from the paper above):

1. The experts may not always benefit from clustering, though it won't
   likely make the predictions worse it may not improve them either.
2. The computational complexity of clustering is non-trivial and may need to be
   avoided depending on the user's needs.

##### Clustered

This approach employs the Birch clustering algorithm implemented in scikit-
learn to cluster the dataset into as many clusters as there will be experts in
the rBCM. The virtue of this is that each expert will have, in the region in
which it is a local expert (now the 'expert' nomenclature is more fitting),
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

## Why not use an rBCM?

There are issues you may encounter that are different from those found when
using a GPR, but they are surmountable with a little careful attention.

If you have multiple experts their predictions will be averaged in some sense,
and if the underlying dataset has significant complexity this averaging may
blur out that complex behavior without careful usage.

For example, highly oscillatory data could result in a rBCM fit, with say only
2 experts, that simply follows the center of that oscillation rather than
tracking along the back and forth motion. A full GPR may capture the entire
motion automatically due to not being affected by that blurring of the two
models. An rBCM captures extreme sudden behaviors in the data worse than a GPR
without careful use.

However, this is easily remedied by having sufficiently large datasets and by
choosing a smarter number of experts. You need enough that each gets a
meaningful sampling along all the complex regions of the dataset, but not so
few that you lose the predictive accuracy resulting from the weighting of the
several semi-informed experts. You want each expert to have data from the
entire upward and downward arc of an oscillatory data set, for example.

Another example is data that looks like an exponential curve; the blurring
effect may cause predictions at the extreme tail to be somewhat smaller than
the full GPR. The averaging from the other models at the lower part of the
curve drags the final weighted prediction down.

It's advisable to fit a full GPR to a subsampling of your data and compare the
output of that with your rBCM to ensure you aren't missing any easily fit
regions. But the exponential curve example is only fixable by messing with the
number of experts.

Similarly, if you are fitting a trivially small dataset, the rBCM is not a
good choice as it will split your already limited data into even smaller
datasets for each model. This package silently serves you a full GPR in the
case that `n < 1024` by implicitly setting the number of experts to 1.

## Relevant References
    
<a name="ref1.">1.</a> [Distributed Gaussian Processes](http://www.jmlr.org/proceedings/papers/v37/deisenroth15.pdf)
<a name="ref2.">2.</a> [Easily digestable powerpoint deck on rBCM's](http://www.doc.ic.ac.uk/~mpd37/talks/2015-05-21-gpws.pdf)
<a name="ref3.">3.</a> [BCM background information](http://www.dbs.ifi.lmu.de/~tresp/papers/bcm6.pdf)
<a name="ref4.">4.</a> [Other alternative ways to scale gaussian process regression](http://www.dbs.ifi.lmu.de/~tresp/papers/nips02_approxgp.pdf)
<a name="ref5.">5.</a> [Generalized Product of Experts for Automatic and Principled Fusion of Gaussian Process Predictions](http://arxiv.org/pdf/1410.7827v2.pdf)

