from math import log10
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import numpy as np
from .utils import create_rng


class BO:
    """
    Sequential Bayesian Optimization framework
    """

    def __init__(self, k, hidden_dim=(100, 10000),
                 spectral_radius=(.9, 1.3), p=(0, 1),
                 alpha=(0, 1), beta=(1e-5, 1e3), random_state=None):
        """

        Parameters
        ----------
        k : tuple
            Range of values for nearest neighbors in small-world network
        hidden_dim : tuple, optional
            Range values for the number of nodes in the reservoir
        spectral_radius : tuple, optional
            Range of values for the spectral radius for the reservoir
        p : tuple, optional
            Range of values to consider for the rewire probability
        alpha : tuple, optional
            Range of values for the leaking rate
        beta : tuple, optional
            Range of values for the L2 regression regularization
        random_state : int or np.random.RandomState, optional
            Random state initializer
        """
        # Check that all the hyper-parameters are tuples with two entries
        # which define the lower and upper bounds for the search space
        hyper_params = [k, hidden_dim, spectral_radius, p, alpha, beta]
        for param in hyper_params:
            assert isinstance(param, tuple), "{} must be a tuple".format(param)
            assert len(param) == 2, "{} must have two arguments; the upper" \
                                    "and lower bound".format(param)

        self.lwr_k = k[0]
        self.upr_k = k[1]
        self.lwr_hidden_dim = hidden_dim[0]
        self.upr_hidden_dim = hidden_dim[1]
        self.lwr_spectral_radius = hidden_dim[0]
        self.upr_spectral_radius = hidden_dim[1]
        self.lwr_p = p[0]
        self.upr_p = p[1]
        self.lwr_alpha = alpha[0]
        self.upr_alpha = alpha[1]
        self.lwr_beta = beta[0]
        self.upr_beta = beta[1]

        self.rng = create_rng(random_state)
        self.gpr = GaussianProcessRegressor(kernel=Matern(),
                                            random_state=self.rng)

        # We need a placeholder for different hyper-parameter values that
        # arrive and the corresponding error values
        self.H = []
        self.y = []

    def update_gpr(self, X, y):
        """
        Updates the Gaussian process with new data and error value

        Updates the Gaussian process by adding, `H`, the list of
        hyper-parameter values that were used with true function and y
        is the resulting error from the model

        Parameters
        ----------
        X : list
            Hyper-parameter values that were tried
        y : float
            Error that resulted from using H on the true function

        Returns
        -------
        None

        """
        self.H.append(X)
        self.y.append(y)

        self.gpr.fit(self.H, self.y)

    def _sample_uniformly(self, num_samples, lwr_bound, upr_bound):
        """
        Samples uniformly from a non-uniform space

        Parameters
        ----------
        num_samples : int
            Number of samples to generate
        lwr_bound : float
            Hyper-parameter lower bound
        upr_bound : float
            Hyper-parameter upper bound

        Returns
        -------
        param_vals : np.ndarray
            Uniformly sampled hyper-parameter values

        """
        # To sample in a uniform fashion we need the base ten representation
        # of the upper and lower bounds and then we treat this as a region
        # to sample
        new_lwr_bound = log10(lwr_bound)
        new_upr_bound = log10(upr_bound)
        samples = self.rng.uniform(low=new_lwr_bound, high=new_upr_bound,
                                   size=(num_samples, 1))
        param_vals = np.power(10, samples)
        return param_vals

    def build_options(self, num_samples=1000):
        """
        Builds matrix which defines possible options for this iteration

        Parameters
        ----------
        num_samples : int, optional
            Number of hyper-parameter samples to generate

        Returns
        -------
        H_space : np.ndarray
            Matrix of options for the ESN hyper-parameters

        """
        k_vals = self.rng.randint(low=self.lwr_k, high=self.upr_k,
                                  size=(num_samples, 1), dtype=np.int32)

        hidden_dim_vals = self.rng.randint(low=self.lwr_hidden_dim,
                                           high=self.upr_hidden_dim,
                                           size=(num_samples, 1),
                                           dtype=np.int32)

        spectral_radius_vals = self.rng.uniform(low=self.lwr_spectral_radius,
                                                high=self.upr_spectral_radius,
                                                size=(num_samples, 1))

        p_vals = self.rng.uniform(low=self.lwr_p, high=self.upr_p,
                                  size=(num_samples, 1))

        alpha_vals = self.rng.uniform(low=self.lwr_alpha, high=self.upr_alpha,
                                      size=(num_samples, 1))

        beta_vals = self._sample_uniformly(num_samples, self.lwr_beta,
                                           self.upr_beta)

        H_space = np.concatenate([k_vals, hidden_dim_vals,
                                  spectral_radius_vals, p_vals, alpha_vals,
                                  beta_vals], axis=1)
        return H_space

    def find_best_choice(self, H_space):
        """
        Finds the best hyper-parameter combination

        Parameters
        ----------
        H_space : np.ndarray
            Matrix of hyper-parameter options

        Returns
        -------
        param_vals : dict
            Best hyper-parameter values for the current Gaussian process

        """

        y_pred = self.gpr.sample_y(H_space, random_state=self.rng)
        best_choice = y_pred.argmax(axis=0)
        hyper_parameters = ['k', 'hidden_dim', 'spectral_radius', 'p', 'alpha',
                            'beta']
        best_vals = H_space[best_choice].flatten()
        param_vals = dict(zip(hyper_parameters, best_vals))
        return param_vals
