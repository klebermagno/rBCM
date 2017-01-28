rBCM Package API
****************


Primary Regressor Class
-----------------------

This is only interface a user need be aware of to get going with basic usage.

It has quite a long name, so we alias it to the following shorter one.

.. autoclass:: rBCM.RBCM
    :show-inheritance:

The following documents the actual class itself.

.. autoclass:: rBCM.rbcm.RobustBayesianCommitteeMachineRegressor
    :members: fit, predict


Weighting Functions
--------------------

There is only one weighting function implemented right now, so it's
not a parameter made available for change on the primary rBCM class.

This should be hidden out of sight during normal usage.

.. autofunction:: rBCM.weighting.differential_entropy_weighting
