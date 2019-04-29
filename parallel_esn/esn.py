import numpy as np
import networkx as nx
from sklearn.metrics import mean_squared_error
from .utils import create_rng, compute_spectral_radius
from .progress import progress


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

        self.YXt = np.zeros((output_dim, 1+input_dim+hidden_dim))
        self.XXt = np.zeros((1+input_dim+hidden_dim, 1+input_dim+hidden_dim))

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

    def _compute_X(self, U):
        """
        Computes the X in Y = W_out X,
        where the column vectors [1;u(n);x(n)] are concatenated
        horizontally over the entire time n = 1,...,T

        Parameters
        ----------
        U : np.ndarray
            Input data array, dimensions N_u x T
            Columns u(n) concatenated horizontally

        Returns
        -------
        X : np.ndarray
            X is [1;u(n);x(n)] concatenated horizontally (n is time) generated
            from input data U.
            Dimensions of (1+ N_u + N_x) x T
        """
        T = U.shape[1]
        Nu = self.input_dim
        Nx = self.hidden_dim
        X = np.ones((1 + Nu + Nx, T))

        # Do first step
        X[1:Nu+1, 0] = U[:, 0]
        xti = np.tanh(self.W_in @ X[:Nu+1, 0])  # can potentially add random init
        X[Nu+1:, 0] = self.alpha * xti             # as a hyper-parameter

        for n in range(1, T):
            X[1:Nu+1, n] = U[:, n]
            xti = np.tanh(self.W_in @ X[:Nu+1, n] + self.W @ X[Nu+1:, n-1])
            X[Nu+1:, n] = (1.-self.alpha)*X[Nu+1:, n-1] + self.alpha*xti
        return X

    def _compute_Wout(self):
        """
        Computes W_out from X (activations) and W
        W_out = YX'(XX' + bI)^(-1)

        Returns
        -------
        W_out : np.ndarray
            Matrix giving Y predicted values from X output from reservoir.

        """
        Id = np.identity(self.XXt.shape[0])
        inner = np.linalg.inv(self.XXt + self.beta*Id)

        self.W_out = np.matmul(self.YXt, inner)

    def train(self, batchU, batchY_true, verbose=1, compute_loss_freq=-1):
        """
        Trains on U's and corresponding Y_true's, batched in first index.

        Parameters
        ----------
        batchU : array_like of np.ndarray
            Batch of input data arrays, columns u(n) concatenated horizontally
            Dimensions - Batch_size x N_x x T_i
        batchY_true : array_like of np.ndarray
            batch of true output data arrays
            Dimensions - Batch_size x N_y x T_i
        verbose : int, optional, default=1
            Whether to print status of training
        compute_loss_freq : int, default=-1
            How often to compute training loss. Only for information, not necessary
            for training. Negative value disables computing training loss.

        Returns
        -------
        loss : np.ndarray
            Returns the loss computed on each sequence where loss was computed, array
            of length ((Batch_size-1) // compute_loss_freq) + 1. None returned if
            compute_loss_freq is less than equal to 0.
        """
        if batchU.shape[0] != batchY_true.shape[0]:
            raise ValueError('batchU and batchY need to have the same first dimension')
        nseq = batchU.shape[0]
        if compute_loss_freq > 0:
            loss = np.zeros((nseq-1) // compute_loss_freq + 1)
        else:
            loss = None
        for s in range(nseq):
            X = self._compute_X(batchU[s, :, :])
            self.XXt += X @ X.T
            self.YXt += batchY_true[s, :, :] @ X.T
            self._compute_Wout()
            status = 'Training:'
            if compute_loss_freq > 0:
                if s % compute_loss_freq == 0:
                    # Use precomputed X instead of recomputing it from U
                    loss[s // compute_loss_freq] = self.score_with_X(X, batchY_true[s, :, :])
                    status += ' loss = {0:.4f}'.format(loss[s // compute_loss_freq])
            if verbose == 1:
                progress(s, nseq, status=status)
        return loss

    def validate(self, batchU, batchY_true, verbose=1):
        """
        Get loss on validation set, given past sequences in batchU and observed outcomes batchY_true

        Parameters
        ----------
        batchU : array_like of np.ndarray
            Batch of input data arrays, columns u(n) concatenated horizontally
            Dimensions - Batch_size x N_x x T_i
        batchY_true : array_like of np.ndarray
            Batch of true output data arrays
            Dimensions - Batch_size x N_y x T_i

        Returns
        -------
        loss : float
            Returns the average of the NMSE losses computed on each sequence
        """
        nseq = batchU.shape[0]
        loss = 0.
        for s in range(nseq):
            curr_loss = self.score(batchU[s, :, :], batchY_true[s, :, :])
            loss += curr_loss
            if verbose == 1:
                progress(s, nseq, status='Validation: loss = {0:.4f}'.format(curr_loss))
        return loss/nseq

    def train_validate(self, trainU, trainY, valU, valY, verbose=1, compute_loss_freq=-1):
        """
        Train on provided training data, and immediately validate
        and return validation loss.

        Parameters
        ----------
        trainU : array_like of np.ndarray
            Batch of training input data arrays, columns u(n) concatenated horizontally.
            Dimensions - Batch_size x N_x x T_i
        trainY : array_like of np.ndarray
            Batch of training true output data arrays.
            Dimensions - Batch_size x N_y x T_i
        valU : array_like of np.ndarray
            Batch of validation input data arrays, columns u(n) concatenated horizontally.
            Dimensions - Batch_size x N_x x T_i
        valY : array_like of np.ndarray
            Batch of validation true output data arrays.
            Dimensions - Batch_size x N_y x T_i
        verbose : int, optional, default=1
            Whether to print status of training
        compute_loss_freq : int, optional, default=-1
            How often to compute training loss. Only for information, not necessary
            for training. Negative value disables computing training loss.

        Returns
        -------
        loss : float
            Returns the sum of the losses computed on each sequence in validation set.
        """
        self.train(trainU, trainY, verbose=verbose, compute_loss_freq=compute_loss_freq)
        return self.validate(valU, valY, verbose=verbose)

    def predict(self, U):
        """
        Predicts Yhat, output observations, given time series of inputs U

        Parameters
        ----------
        U : np.ndarray
            Input data array, columns u(n) concatenated horizontally.
            Dimensions - N_u x T

        Returns
        -------
        Yhat : np.ndarray
            Prediction of observations. Returns feature vectors as columns stacked horizontally
            in time. Take the transpose of this output to
            obtain feature vectors as rows stacked vertically in time.
        """
        if not isinstance(self.W_out, np.ndarray):  # Check if W_out exists yet
            raise UnboundLocalError('Must train network before predictions can be made.')
        W_out = self.W_out
        if U.shape[0] != self.input_dim:
            raise ValueError("Dimension error: provided U input data array does not have "
                             "the same number of features as the input data it was trained "
                             "on.")
        X = self._compute_X(U)
        Yhat = np.matmul(W_out, X)
        return Yhat

    def predict_with_X(self, X):
        """
        Predicts Yhat, output observations, given X already generated
        from input data U.

        Parameters
        ----------
        X : np.ndarray
            X is [1;u(n);x(n)] concatenated horizontally (n is time), generated
            from the input data U.
            Dimensions of (1+ N_u + N_x) x T

        Returns
        -------
        Yhat : np.ndarray
            Prediction of observations.
        """
        if not isinstance(self.W_out, np.ndarray):  # Check if W_out exists yet
            raise UnboundLocalError('Must train network before predictions can be made.')
        W_out = self.W_out
        Yhat = np.matmul(W_out, X)
        return Yhat

    def score(self, U, Y_true):
        """
        Computes loss

        Parameters
        ----------
        U : np.ndarray
            Input data array, columns u(n) concatenated horizontally.
            Dimensions - N_u x T
        Y_true : np.ndarray
            Target output array,  y(n) concatenated horizontally in time.
            Dimensions - N_y x T

        Returns
        -------
        error : float
            Normalized root mean square error (NRMSE). Each feature's NRMSE is
            computed separately and averaged together at the end.

        """
        Yhat = self.predict(U)
        num_features = Y_true.shape[0]
        error = 0.
        for j in range(num_features):
            var = np.var(Y_true[j, :])
            if var == 0.:
                var = 1.
            error += np.sqrt(mean_squared_error(Y_true[j, :], Yhat[j, :])
                             / var)
        return error/num_features

    def score_with_X(self, X, Y_true):
        """
        Computes loss given input data X already processed with reservoir
        activations. Functions identically to running:

        self.score(self._compute_X(U), Y_true)

        if provided X corresponds to U.

        Parameters
        ----------
        X : np.ndarray
            X is [1;u(n);x(n)] concatenated horizontally (n is time), generated
            from the input data U.
            Dimensions of (1+ N_u + N_x) x T
        Y_true : np.ndarray
            Target output array,  y(n) concatenated horizontally in time.
            Dimensions - N_y x T

        Returns
        -------
        error : float
            Normalized mean square error (NRMSE). Each feature's NRMSE is
            computed separately and averaged together at the end.

        """
        Yhat = self.predict_with_X(X)
        num_features = Y_true.shape[0]
        error = 0.
        for j in range(num_features):
            var = np.var(Y_true[j, :])
            if var == 0.:
                var = 1.
            error += np.sqrt(mean_squared_error(Y_true[j, :], Yhat[j, :])
                             / var)
        return error/num_features
