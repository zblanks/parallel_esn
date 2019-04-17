import numpy as np
import networkx as nx
from .utils import create_rng, compute_spectral_radius


class ESN:
    """
    Sequential Echo State Network class

    Parameters
    ----------
    input_dim : int
        Size of input dimension
    hidden_dim : int
        Number of hidden units in the W matrix
    output_dim : int
        Dimensionality of the output
    k : int
        k-nearest neighbors each node is connected to in the small-world
        network for the hidden layer
    spectral_radius : float, optional
        Spectral radius of the reservoir
    p : float, optional
        Re-wiring probability for small-world network
    beta : float, optional
        Regularization parameter for L2 regression
    alpha : float, optional
        ESN leaking rate
    random_state : int or np.random.RandomState, optional
        Random state initializer
    weight_distn : {"uniform", "normal"}, optional
        Distribution of reservoir weights
    """

    def __init__(self, input_dim, hidden_dim, output_dim, k,
                 spectral_radius=0.9, p=0.1, beta=1e-3, alpha=0.7,
                 random_state=None, weight_distn='uniform'):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.k = k
        self.spectral_radius = spectral_radius
        self.p = p
        self.beta = beta
        self.alpha = alpha
        self.rng = create_rng(random_state)
        self.weight_distn = weight_distn

    def _initialize_hidden_layer(self):
        """
        Defines the hidden layer matrix as a small world network

        Returns
        -------
        W : np.ndarray
            Hidden layer weight matrix

        """
        G = nx.watts_strogatz_graph(self.hidden_dim, self.k, self.p, self.rng)
        A = nx.to_numpy_array(G)

        if self.weight_distn == 'uniform':
            weights = self.rng.uniform(low=-1, high=1, size=(self.hidden_dim,
                                                             self.hidden_dim))
        elif self.weight_distn == 'normal':
            weights = self.rng.randn(d0=self.hidden_dim, d1=self.hidden_dim)
        else:
            raise ValueError('Only support "uniform" or "normal" weights')

        W = A * weights

        # The spectral radius can be controlled by normalizing the desired
        # radius over the current value for the matrix
        current_radius = compute_spectral_radius(W)
        if current_radius == 0.:
            raise ValueError('Spectral radius equals zero; make graph less '
                             'sparse')

        W *= self.spectral_radius / current_radius
        return W.astype(np.float32)

    def _initialize_input_layer(self):
        """
        Defines the input layer matrix

        Returns
        -------
        W_in : np.ndarray
            Dense weight matrix with weights drawn from provided distribution

        """
        if self.weight_distn == 'uniform':
            return self.rng.uniform(low=-1, high=1,
                                    size=(self.hidden_dim,
                                          (1 + self.input_dim)))
        else:
            return self.rng.randn(d0=self.input_dim, d1=(1 + self.input_dim))
