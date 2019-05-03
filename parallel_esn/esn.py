import warnings
import numpy as np
import networkx as nx
from sklearn.metrics import mean_squared_error
from .utils import create_rng, compute_spectral_radius
from .progress import progress
from .train_esn import compute_X


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
                 random_state=None, weight_distn='uniform', use_cython=True):
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
        use_cython : bool, optional
            Whether to use the Cython compiled code when computing the X matrix
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
        self.use_cython = use_cython
        self.W_out = None
        self.W = self._initialize_hidden_layer()
        self.W_in = self._initialize_input_layer()

        self.X0 = np.zeros((1+input_dim+hidden_dim))

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
        return W.astype(np.float64)

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

        X calculation is started from the saved reservoir state X0,
        which is essentially the -1th column of X, i.e. the time right
        before t = 0.

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
        if self.use_cython:
            X = compute_X(U.astype(np.float64), self.W_in, self.W, self.alpha,
                          self.input_dim, self.hidden_dim, self.X0)
        else:
            T = U.shape[1]
            Nu = self.input_dim
            Nx = self.hidden_dim
            X = np.ones((1 + Nu + Nx, T))

            # Do first step. It computes the first column of X based off of
            # the existing reservoir state X0, which is just zero if the reservoir
            # is not warmed up, or the state from the last training/prediction if
            # it is.
            X[1:Nu+1, 0] = U[:, 0]
            xti = np.tanh(self.W_in @ X[:Nu+1, 0] + self.W @ self.X0[Nu+1:])
            X[Nu+1:, 0] = (1.-self.alpha)*self.X0[Nu+1:] + self.alpha * xti

            for n in range(1, T):
                X[1:Nu+1, n] = U[:, n]
                xti = np.tanh(self.W_in @ X[:Nu+1, n] + self.W @ X[Nu+1:, n-1])
                X[Nu+1:, n] = (1.-self.alpha)*X[Nu+1:, n-1] + self.alpha*xti

        # Save final reservoir state
        self.X0 = X[:, -1]
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

    def clear_state(self):
        """
        Clears X0, the reservoir's memory of previous inputs and neural state
        """
        self.X0 = np.zeros((1+self.input_dim+self.hidden_dim))

    def reset(self):
        """
        Reset output layer training and clear state
        """
        self.W_out = None
        self.YXt = np.zeros((self.output_dim, 1+self.input_dim+self.hidden_dim))
        self.XXt = np.zeros((1+self.input_dim+self.hidden_dim, 1+self.input_dim+self.hidden_dim))
        self.clear_state()

    def train(self, batchU, batchY_true, clear_state=False, warmup=10, verbose=1,
              compute_loss_freq=-1):
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
        clear_state : boolean, optional, default=False
            Whether to clear the reservoir memory in between batches. If False, the training
            on the batches is equivalent to if the batches were all concatenated into a single
            time series.
        warmup : int, optional, default=10
            The number of states to discard at the beginning of training, before initial
            transients in the reservoir have died out. The amount to discard depends on
            the memory of the network and typically ranges from 10s to 100s. If batches are
            to be treated as independent, with clear_state=True, warmups can typically be shorter
            since the zeroed reservoir initialization would be the normal operating mode of
            the ESN.
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
            if clear_state:
                self.clear_state()
            X = self._compute_X(batchU[s, :, :])
            if warmup > 0:  # Need to discard more states
                if warmup >= X.shape[1]:
                    if clear_state:
                        warnings.warn("Warning: specified to discard more warmup states than "
                                      "there are in the batch of input data. Network is not "
                                      "being trained.")
                    elif s == nseq - 1:
                        warnings.warn("Warning: specified to discard more warmup states than "
                                      "there are input data. Network was not trained.")
                    warmup -= X.shape[1]  # ignore all X in this batch, decrease warmup count
                else:
                    self.XXt += X[:, warmup:] @ X[:, warmup:].T
                    self.YXt += batchY_true[s, :, warmup:] @ X[:, warmup:].T
                    warmup = 0  # Done with warmup
            else:  # Done with warmup
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

    def validate(self, batchU, batchY_true, warmup=10, verbose=1):
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
        warmup : int, optional, default=10
            The number of states to discard at the beginning of each validation batch, before initial
            transients in the reservoir have died out. The amount to discard depends on
            the memory of the network and typically ranges from 10s to 100s.
        verbose : int, optional, default=1
            Whether to print status of training

        Returns
        -------
        loss : float
            Returns the average of the NMSE losses computed on each sequence
        """
        nseq = batchU.shape[0]
        loss = 0.
        if warmup >= batchU.shape[2]:
            warmup = 0
            warnings.warn("Warning: Validation batch size is too small for specified "
                          "warm-up time. Warm-up set to zero")
        for s in range(nseq):
            curr_loss = self.score(batchU[s, :, warmup:], batchY_true[s, :, warmup:])
            loss += curr_loss
            if verbose == 1:
                progress(s, nseq, status='Validation: loss = {0:.4f}'.format(curr_loss))
        return loss/nseq

    def recursive_validate(self, batchU, batchY_true, input_len, pred_len, verbose=1):
        """
        Get loss on validation set for recursive one-step prediction. Uses recursive_score
        to compute the total error.

        Parameters
        ----------
        batchU : array_like of np.ndarray
            Batch of input data arrays, columns u(n) concatenated horizontally
            Dimensions - Batch_size x N_x x T_i
        batchY_true : array_like of np.ndarray
            Batch of true output data arrays
            Dimensions - Batch_size x N_y x T_i
        input_len : int
            The input length to be fed to the ESN before recursive single-step prediction.
        pred_len : int
            The number of predictions desired.
        verbose : int, optional, default=1
            Whether to print status of training

        Returns
        -------
        loss : float
            Returns the average of the NMSE losses computed on each sequence
        """
        nseq = batchU.shape[0]
        loss = 0.
        for s in range(nseq):
            curr_loss = self.recursive_score(batchU[s, :, :], batchY_true[s, :, :],
                                             input_len, pred_len)
            loss += curr_loss
            if verbose == 1:
                progress(s, nseq, status='Validation: loss = {0:.4f}'.format(curr_loss))
        return loss/nseq

    def train_validate(self, trainU, trainY, valU, valY,
                       warmup=10, verbose=1, compute_loss_freq=-1):
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
        warmup : int, optional, default=10
            The number of states to discard at the beginning of each train/validation batch,
            before initial transients in the reservoir have died out. The amount to discard
            depends on the memory of the network and typically ranges from 10s to 100s.
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
        self.train(trainU, trainY, warmup=warmup,
                   verbose=verbose, compute_loss_freq=compute_loss_freq)
        return self.validate(valU, valY, warmup=warmup, verbose=verbose)

    def recursive_train_validate(self, trainU, trainY, valU, valY,
                                 input_len, pred_len,
                                 warmup=10, verbose=1, compute_loss_freq=-1):
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
        input_len : int
            The input length to be fed to the ESN before recursive single-step prediction.
        pred_len : int
            The number of predictions desired.
        warmup : int, optional, default=10
            The number of states to discard at the beginning of each train/validation batch,
            before initial transients in the reservoir have died out. The amount to discard
            depends on the memory of the network and typically ranges from 10s to 100s.
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
        self.train(trainU, trainY, warmup=warmup,
                   verbose=verbose, compute_loss_freq=compute_loss_freq)
        return self.recursive_validate(valU, valY, input_len, pred_len, verbose=verbose)

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
        from input data U. Useful if X has already been computed and a prediction
        is desired without affecting the current state of the reservoir.

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

    def recursive_predict(self, U, iterations, cold_start=False):
        """
        Predicts Yhat, output observations following the given time series of inputs U
        This method assumes that the network has been trained to produce one-step-forecasting,
        where the output y(t) corresponds to u(t+1), i.e. what the next input would have been.
        Currently this method only supports predicting all features provided as input; the
        dimensions of vector y(t) must match the dimensions of u(t).

        For the first step, observed values u(t) for t=0..T-1 are used to produce the first
        predicted value y(t) = hat{u}(t+1), which is then fed back to the network as an
        input in order to produce y(t+1). This recursion is continued for the specified
        number of iterations.

        Parameters
        ----------
        U : np.ndarray
            Input data array, columns u(n) concatenated horizontally.
            Dimensions - N_u x T
        iterations : int
            How many future times to predict.
        cold_start : boolean, optional, default=False
            Whether to clear reservoir state before driving the reservoir with input data U.
            If the input data follows directly after training data, a warm start is sensible.
            However, if the provided data is temporally disconnected from the training data,
            a cold start could be useful for reproducibility if this method will be called
            multiple times, on the same data or on other inputs.

        Returns
        -------
        Yhat : np.ndarray
            Prediction of observations. Returns feature vectors as columns stacked horizontally
            in time. Take the transpose of this output to
            obtain feature vectors as rows stacked vertically in time.
        """
        if U.shape[0] != self.output_dim:
            raise ValueError("Method recursive_predict requires that input_dim = output_dim")
        if cold_start:
            self.clear_state()
        X = self._compute_X(U)
        Yhat = np.zeros((self.output_dim, iterations))
        Ynext = self.W_out @ X[:, -1:]  # Maintain 2D shape
        Yhat[:, 0:1] = Ynext
        for i in range(1, iterations):
            Xpresent = self._compute_X(Ynext)
            Ynext = self.W_out @ Xpresent[:, 0:]  # Maintain 2D shape
            Yhat[:, i:i+1] = Ynext
        return Yhat

    def recursive_score(self, U, Y_true, input_len, pred_len):
        """
        Computes loss in recursive one-step prediction, intended for validation set.

        Parameters
        ----------
        U : np.ndarray
            Input data array, columns u(n) concatenated horizontally.
            Dimensions - N_u x T
        Y_true : np.ndarray
            Target output array,  y(n) concatenated horizontally in time.
            Dimensions - N_y x T
        input_len : int
            The input length to be fed to the ESN before recursive single-step prediction.
        pred_len : int
            The number of predictions desired.

        Returns
        -------
        error : float
            Normalized mean square error (NMSE). On the prediction. Each feature's
            NMSE is computed separately and averaged together at the end.

        """
        if U.shape[1] < input_len + pred_len:
            raise ValueError("Method recursive_score requires that U has at least input_len + "
                             "pred_len + 1 points in time. U only has {} time points"
                             .format(U.shape[1]))
        Yhat = self.recursive_predict(U[:, :input_len], pred_len)
        num_features = Y_true.shape[0]
        error = 0.
        for j in range(num_features):
            strt = input_len - 1
            end = strt + pred_len
            var = np.var(Y_true[j, strt:end])
            if var == 0.:
                var = 1.
            error += mean_squared_error(Y_true[j, strt:end], Yhat[j, :]) / var
        return error/num_features

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
            Normalized mean square error (NMSE). Each feature's NMSE is
            computed separately and averaged together at the end.

        """
        Yhat = self.predict(U)
        num_features = Y_true.shape[0]
        error = 0.
        for j in range(num_features):
            var = np.var(Y_true[j, :])
            if var == 0.:
                var = 1.
            error += mean_squared_error(Y_true[j, :], Yhat[j, :]) / var
        return error/num_features

    def score_with_X(self, X, Y_true):
        """
        Computes loss given input data X already processed with reservoir
        activations. Functions identically to running:

        self.score(self._compute_X(U), Y_true)

        provided X corresponds to U, and if starting from the same reservoir
        initial state.

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
            Normalized mean square error (NMSE). Each feature's NRMSE is
            computed separately and averaged together at the end.

        """
        Yhat = self.predict_with_X(X)
        num_features = Y_true.shape[0]
        error = 0.
        for j in range(num_features):
            var = np.var(Y_true[j, :])
            if var == 0.:
                var = 1.
            error += mean_squared_error(Y_true[j, :], Yhat[j, :]) / var
        return error/num_features
