=====================
Bayesian Optimization
=====================

Overview
--------
The way to access the Bayesian optimization (BO) code used to help search the
hyper-parameter space of an echo state network (ESN) is via the
:func:`~parallel_esn.bo.BO` class. This allows one define ranges for the
relevant hyper-parameters for the ESN model and provides functionality to
update the underlying Gaussian process.

Function Documentation
----------------------
.. autoclass:: parallel_esn.bo.BO
    :members:
