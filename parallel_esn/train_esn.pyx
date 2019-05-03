import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_X(np.ndarray[np.float64_t, ndim=2] U,
                np.ndarray[np.float64_t, ndim=2] W_in,
                np.ndarray[np.float64_t, ndim=2] W, double alpha, int input_dim,
                int hidden_dim, np.ndarray[np.float64_t, ndim = 1] X0):

    cdef:
        int T = U.shape[1]
        int Nu = input_dim
        int Nx = hidden_dim

        np.ndarray[np.float64_t, ndim=2] X = np.ones(shape=(1 + Nu + Nx, T),
                                                     dtype=np.float64)

        np.ndarray xti = np.ones(shape=(Nx, 1), dtype=np.float64)

        int n

    # Do the first step
    X[1:(Nu + 1), 0] =  U[:, 0]
    xti = np.tanh(W_in @ X[:(Nu + 1), 0] + W @ X0[(Nu + 1):])
    X[(Nu + 1):, 0] = (1.-alpha)*X0[Nu+1:] + (alpha * xti)

    for n in range(1, T):
        X[1:(Nu + 1), n] = U[:, n]
        xti = np.tanh((W_in @ X[:(Nu + 1), n]) + (W @ X[(Nu + 1):, n-1]))
        X[(Nu + 1):, n] = ((1. - alpha) * X[(Nu + 1):, n-1]) + (alpha * xti)

    return X
