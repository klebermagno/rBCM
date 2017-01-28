# rBCM - robust Bayesian Committee Machine

A python package for fitting a robust Bayesian Committee Machine (rBCM) to data
and then predicting with it. This is potentially more scalable than Gaussian
Process Regression (GPR) while still providing a good approximation to a full
GPR model. The rBCM contains the GPR model as a special case.

Other alternatives to scaling GPR have some undesirable downsides (discussed in
reference 4). They are typically either too inaccurate to a full GPR or have
worse scalability characteristics than the rBCM.

The paper below was the primary basis for this implementation.

* [Distributed Gaussian Processes](http://www.jmlr.org/proceedings/papers/v37/deisenroth15.pdf)

## Installation

Install the package into a virtual environment from source with the following:

```
sudo pip install virtualenv
pip virtualenv venv
source ./venv/bin/activate
python setup.py install
```

You can also install it without a virtual environment with just `python
setup.py install`.

#### Installing Dev Tools

The utilities for testing and building the documentation are not included in
the base installation. You can install them with `pip install -r
requirements-dev.txt`

## Test

After installing the dev tools, you can run the test suite with `tox`.

## Documentation

There exists some documentation, but you have to build it yourself first with
Sphinx for now.

You can do this in the `doc` directory with the `Makefile`. Use the command
`make help` there to see what formats are available.

If you call `make html` there will be a file at the location
`docs/build/html/index.html` produced which you can open with your browser to
see the full docs.

## How does it work?

The general strategy is to fit many smaller GPR expert models on subsets of the
data and then take what amounts to a complicated weighted average of their
predictions.

Using their predictive variance to find a single prediction that is a
good-enough approximation of a full GPR model over the entire dataset.

This is possible due to the GPR model providing an estimate of its predictive
variance. Each expert only has some small fraction of the data and will have
high predictive variance in most places, but quite low variance in places where
its data subset happens to dominate.

By weighting predictions of models by their certainty, we let experts who know
a lot about a particular region of the sample space have more highly weighted
predictions in those places and experts who know little make low impact
guesses.

Of course, this is a tradeoff; you get averaged, 'blurred' predictions in many
cases.

## Why is it faster?

This model offers two performance benefits:

* The rBCM reduces the impact of the poor GPR scaling, because each expert only
  handles some subset of the total data.
* The rBCM opens up parallelism during both training and fitting stages - each
  expert can be trained and predicted with independently and then simply
  weighted together.

## What are the problems with it?

Though this approach has its benefits, it quickly introduces a number of open
problems:

* What is the correct number/density of experts for the size of my dataset? Or
  equivalently, how many points per expert can we tolerate for my computational
  and predictive accuracy needs? Is the best we can offer simply, 'up to you'?
* What is the best way to distribute the data to experts? Randomly? In clusters?
* What are the obvious degenerate datasets where the rBCM dramatically fails to
  approximate the GPR when it really ought not to?
* Are there ways to intelligently distribute points to the experts unevenly by
  abusing factors such as density of the samples in the space?
* This package simply degenerates to a single full GPR for trivially small
  datasets, you won't save any meaningful time by using a rBCM on something
  that small and only stand to lose accuracy. What are better rules for
  identifying when using a rBCM will hurt you more than help?

## Usage

The following are different considerations to be aware of when using this
package. Though you may easily use the package without looking at any of these
things, here is an example usage:

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

Currently, an rBCM is passed a single kernel and that is used as the kernel for
all the experts. This must be one of the kernel subclasses from the
`scikit-learn.gaussian_process.kernels` module. This is a choice of prior and
you may pass any you like, but know that each expert's parameters get optimized
independently of one another against only the data partitioned to that expert.

### Batching

In the prediction step, you can pass a `batch_size` parameter to instruct the
experts to predict at only `batch_size` data locations in parallel at once.
This can be used to limit maximum system memory consumption. Prediction with
GPR models is known to be a memory hog.

Though the default of `batch_size=-1` sets there to be no batching by default
and you may not need it depending on how many predictions you request at once.

### Partitioning

This package currently supports two ways to partition the data set among
experts. Which one is used is passed as a boolean argument at instantiation
time of the rBCM object with the `locality` parameter. If False, the random
approach is taken. If True, a pre-processing step clusters the dataset and then
provides each expert with one cluster as it's partition.

##### Random

The first is simply a random partitioning of the data into as many equally-
sized chunks as there will be experts. This may be desirable for two reasons
(which I have taken from the paper above):

1. The experts may not always benefit from clustering, though it won't
   likely make the predictions worse it may not improve them either.
2. The computational complexity of clustering is non-trivial and may need to be
   avoided in some cases.

##### Clustered

This approach employs the Birch clustering algorithm implemented in scikit-
learn to cluster the dataset into as many clusters as there will be experts in
the rBCM. The virtue of this is that each expert will have, in the region in
which it is a local expert (now the 'expert' nomenclature is more fitting),
all the possible data for that location. It will not have some subsampling of
the data in that region given by random chance, it will have all of it.

This hopefully leads to the weighting procedure finding only one expert at each
prediction location which has very high certainty or perhaps just a few sharing
some middling confidence, while all the other models should have very low
certainty.

### Weighting

The core idea that enables the rBCM is that a Gaussian Process regression
model can provide a measure of uncertainty for its predictions, not just the
value of the prediction itself. Not all statistical models do this, and as
such this approach of divying up data into experts and then merging them is
not always applicable when attempting to use other models as the underlying
experts.

The exact weighting formula is given in the paper at the top of this readme on
page 5 and involves first computing a weight `beta_k` for each prediction
location for each `expert_k`. It's not obvious what uncertainty metric is best
to use and several models exist whose only distinction is which metric was
chosen, but the following paper makes some suggestions in section 2.3:

* [Generalized Product of Experts for Automatic and Principled Fusion of Gaussian Process Predictions](http://arxiv.org/pdf/1410.7827v2.pdf)

This package currently only supports one choice of beta metric for now. The
beta is taken to be half the difference in the log of the variance of the
posterior from the log of the variance of the prior. This is also known as the
difference in differential entropy between the posterior and prior.

In pseudocode, for a single prediction location `x` and a single `expert_k` the value
of `beta_k` is given as:

```python
beta_k = (0.5) * (log(prior_variance) - log(posterior_variance(expert_k, x)))
```

Note that the `prior_variance` is a single constant value (because all our
experts are the same underlying GPR model with the same kernel they have the
same `prior_variance`), while `posterior_variance(expert_k, x)` depends on the
specific expert and location of the prediction because they all are trained on
different data.

This `beta_k` term is then used as a scaling factor when computing the merged
prediction from all the expert's predictions as well as when computing the
variance of the overall rBCM's predictions.

## Relevant References

<a name="ref1.">1.</a> [Distributed Gaussian Processes](http://www.jmlr.org/proceedings/papers/v37/deisenroth15.pdf)

<a name="ref2.">2.</a> [Easily digestable powerpoint deck on rBCM's](http://www.doc.ic.ac.uk/~mpd37/talks/2015-05-21-gpws.pdf)

<a name="ref3.">3.</a> [BCM background information](http://www.dbs.ifi.lmu.de/~tresp/papers/bcm6.pdf)

<a name="ref4.">4.</a> [Other alternative ways to scale gaussian process regression](http://www.dbs.ifi.lmu.de/~tresp/papers/nips02_approxgp.pdf)

<a name="ref5.">5.</a> [Generalized Product of Experts for Automatic and Principled Fusion of Gaussian Process Predictions](http://arxiv.org/pdf/1410.7827v2.pdf)

