===========================
Echo State Network Training
===========================

Overview
--------
The primary interface with the echo state network (ESN) in this code base is
through the :func:`~esn.esn.ESN` class. This class allows one to define an
ESN and fit it to data with a given set of hyper-parameters.

Defining the Network
--------------------
The usual format one uses when defining an ESN is building a sparse matrix
with a specified spectral radius, :math:`\rho`, that follows either a uniform
or a normal distribution for the weights. However, we have opted to define
the reservoir of the ESN as a small-world network. We did this because there
has been recent research which suggests that by having a more fine-grained
specification of the reservoir network, one can see improvements in
out-of-sample performance for the model :cite:`kawai2017echo`.


Thus instead of specifying the sparsity of the reservoir, a user can now have
greater control over the network by providing the number of neighbors and the
re-wiring probability for the graph.

Function Documentation
----------------------

.. autoclass:: parallel_esn.esn.ESN
    :members:

.. bibliography:: references.bib
