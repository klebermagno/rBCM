Introduction to rBCM
====================

This package implement a robust Bayesian Committee Machine as described in
`this paper <https://arxiv.org/abs/1502.02843>`_ from the proceedings of ICML
2015.

WARNINGS
--------

**Alpha software! Definitely still broken and working on it.**

Description
-----------

From the abstract of that paper:

    "A robust Bayesian Committee Machine (rBCM) is a scalable product-of-experts
    statistical model for large-scale distributed Gaussian process regression
    (GPR)."

See :doc:`here </references/>` for some reference papers and other resources on
the model. It's primary advantage over GPR is scalability, while avoiding
succumbing to some of the downsides of alternative GPR scaling mechanisms.
Namely an unacceptably large increase in predictive variance in some
circumstances or too large of deviations from the base GPR model.

This Python package implements the rBCM on top of `sklearn`_'s implementation
of GPR. This package does not implement the distributed execution the rBCM
could support, but does offer a multiprocessing single-machine implementation
that examples the performance benefits and statistical behavior of the rBCM as
compared to the underlying GPR. One could hopefully swap out the underlying GPR
implementation without too much work.

Here is an quick example of using this package:

.. code-block:: python

    import numpy as np
    from rBCM.rbcm import RobustBayesianCommitteeMachineRegressor as RBCM

    # The function we will generate data from
    def underlying_function(x):
        return np.abs(np.sin(x)) * 5

    # Generate an artificial sample dataset
    X = np.linspace(10, 20, 100).reshape(-1, 1)
    y = underlying_function(X)

    # Fit the dataset and predict at the same points
    machine = RBCM()
    machine.fit(X, y)
    predictions = machine.predict(X)

You should look at the `Jupyter notebooks
<https://github.com/lucaskolstad/rBCM/tree/master/notebooks/>`_ for the package
to see more examples of usage, examples with plots, and comparisons to the GPR.

.. _sklearn: http://http://www.scikit-learn.org/stable/

Installation
-------------

You can install the rBCM package from source using setuptools with the
following commands:

.. code-block:: bash

    git clone https://github.com/lucaskolstad/rBCM
    python rBCM/setup.py install

You can also install rBCM from pypi using pip:

.. code-block:: bash

    pip install rBCM

The dependencies for building documentation and running tests are separated
into the file `requirements-dev.txt` and can be installed with:

.. code-block:: bash

    pip install -r requirements-dev.txt


Building Documentation
----------------------

This package uses Sphinx for documentation. The Makefile in the :code:`docs/`
directory can be used to build the documentation after installing the dev
dependencies.

Use :code:`make help` to see the available options. Try :code:`make html` to
build the documentation into :code:`docs/_build/html` and open
:code:`docs/_build/html/index.html` with your browser to view it.

Testing
--------------

This package uses :code:`pytest` and :code:`tox` to do testing. Run :code:`tox`
in the top-level directory to run the tests after installing the dev
dependencies.
