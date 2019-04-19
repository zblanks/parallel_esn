import numpy as np
import networkx as nx
from sklearn.metrics import mean_squared_error
from .utils import create_rng, compute_spectral_radius


class ESN:
    """
    Sequential Echo State Network class

    Parameters
    ----------
    input_dim : int
        Size of input dimension, N_u
    hidden_dim : int
        Number of hidden units in the W matrix, N_x
    output_dim : int
        Dimensionality of the output, N_y
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
        """

        Parameters
        ----------
        input_dim : int
            Size of input dimension, N_u
        hidden_dim : int
            Number of hidden units in the W matrix, N_x
        output_dim : int
            Dimensionality of the output, N_y
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
        self.W_out = None
        self.W = self._initialize_hidden_layer()
        self.W_in = self._initialize_input_layer()

        self.YXt = np.zeros((output_dim,input_dim))
        self.XXt = np.zeros((input_dim,input_dim))

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

    def _compute_X(self,U):
        """
        Computes the X in Y = W_out X,
        where the column vectors [1;u(n);x(n)] are concatenated
        horizontally over the entire time n = 1,...,T

        Parameters
        ----------
        U : np.ndarray, dimensions N_u x T
            input data array, columns u(n) concatenated horizontally

        Returns
        -------
        X : np.ndarray, dim (1+ N_u + N_x) x T
            X is [1;u(n);x(n)] concatenated horizontally (n is time)
        """
        T = U.shape[1]
        Nu = self.input_dim
        Nx = self.hidden_dim
        X = np.ones((1+ Nu + Nx,T))

        # Do first step
        X[1:Nu+1,0] = U[:,0]
        xti = np.tanh(self.W_in @ X[:Nu+1,0]) # can potentially add random init 
        X[:,0] = self.alpha * xti             # as a hyper-parameter

        for n in range(1,T):
            N[1:Nu+1,n] = U[:,n]
            xti = np.tanh(self.W_in @ X[1:Nu+1,n] + self.W @ X[Nu:,n-1])
            X[Nu:,n] = (1.-self.alpha)*X[Nu:,n-1] + self.alpha*xti
        
        return X

    def _compute_Wout(self, Y_true):
        """
        Computes W_out from X_out (activations) and W
        W_out = YX'(XX' + bI)^(-1)

        Parameters
        ----------
        Y_true : np.ndarray, dimensions N_y x T
            Target output array,  y(n) concatenated horizontally in time

        Returns
        -------
        W_out : np.ndarray

        """
        shape = X_out.shape[0]
        Id = np.identity(shape)
        inner = np.linalg.inv(np.matmul(X_out, X_out.T) + self.beta*Id)
        outer = np.matmul(Y_true, X_out.T)

        self.W_out = np.matmul(outer, inner)

    def train(self, batchU, batchY_true):
        """
        Trains on U's and corresponding Y_true's, batched in rows.

        Parameters
        ----------
        batchU : list of np.ndarray, dimensions Unknown x N_x x T_i
            batch of input data arrays, columns u(n) concatenated horizontally
        batchY_true : list of np.ndarray, dimensions Unknown x N_y x T_i
            batch of true output data arrays

        Returns
        -------
        >>> Perhaps training loss?
        """

    def predict(self, U):
        """
        Predicts Yhat, output observations, given time series of inputs U

        Parameters
        ----------
        U : np.ndarray, dimensions N_u x T
            input data array, columns u(n) concatenated horizontally

        Returns
        -------
        Yhat : np.ndarray
            Prediction of observations

        """
        W_out = self.W_out
        X = self._compute_X(U)
        Yhat = np.matmul(W_out, X_out)
        return Yhat

    def score(self, U, Y_true):
        """
        Computes W_out from X_out (activations) and W
        W_out = YX'(XX' + bI)^(-1)

        Parameters
        ----------
        U : np.ndarray, dimensions N_u x T
            input data array, columns u(n) concatenated horizontally
        Y_true : np.ndarray, dimensions N_y x T
            Target output array,  y(n) concatenated horizontally in time

        Returns
        -------
        error : float
        >>> Check this section; will mean squared error handle matrices?

        """
        Yhat = self.predict(U)
        return mean_squared_error(y_true, yhat)


