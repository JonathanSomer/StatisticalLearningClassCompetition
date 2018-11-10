import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from regressors.base_regressor import BaseRegressor
from data_pre_processing.clean_data import remove_bad_movies
from data_pre_processing.fill_missing_values import fill_ratings_with_mean_per_user
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


class KNNRegressor(BaseRegressor):

    def __init__(self):
        self._reg = KNeighborsRegressor(n_neighbors=50, weights='distance')
        self.bad_movie_indexes = None
        self.scaler = MinMaxScaler()
        self.pca = None

    def fit(self, X, Y):
        assert len(X.shape) == 3
        X = self._prepare_X(X, Y, train = True)
        self._reg = self._reg.fit(X, Y[:, 1])

    def predict(self, X):
        assert len(X.shape) == 3
        X = self._prepare_X(X)
        return self._reg.predict(X)

    def __str__(self):
        return "knn"


    def _prepare_X(self, X_raw, Y=None, train = False):
        X_raw, self.bad_movie_indexes = remove_bad_movies(X_raw, self.bad_movie_indexes)
        X, _ = fill_ratings_with_mean_per_user(X_raw)
        X = X[:,1,:] # only ratings

        if train:
            self.scaler.fit(X)
        X_rescaled = self.scaler.transform(X)

        if train:
            self.pca = PCA(n_components = 75).fit(X_rescaled)
        X_components = self.pca.transform(X_rescaled)
        
        return X_components.reshape(X.shape[0], -1)

# def applyA(A, vec):
#     #return np.matmul(A, vec)
#     result = np.zeros(vec.shape)
#     for i in range(A.shape[0]):
#         result[i] = A[i, i] * vec[i]
#     return result

# def compute_softmax_norm_i(A, inp, i):
#     softmax_norm = 0.
#     for k in range(inp.shape[0]):
#         if i == k:
#             continue
#         exponent = applyA(A, inp[i]) - applyA(A, inp[k])
#         exponent = np.dot(exponent, exponent)
#         softmax_norm += np.exp(-exponent)
#     return softmax_norm
        

# def compute_pij(A, inp, i, j):
#     if i == j: 
#         return 0 # since pij == 0
#     exponent = applyA(A, inp[i]) - applyA(A, inp[j])
#     exponent = np.dot(exponent, exponent)
#     pij = np.exp(-exponent) / compute_softmax_norm_i(A, inp, i)
#     return pij

# def nca(A, inp, label, lr=0.5):
#     inp = transform(A, inp)
#     for i in range(inp.shape[0]):
#         p = 0.
#         for j in range(inp.shape[0]):
#             if label[i] == label[j]:
#                 p += compute_pij(A, inp, i, j)
       
#         #print 'p=',p
#         first_term = np.zeros( (inp.shape[1], inp.shape[1]) )
#         second_term = np.zeros( (inp.shape[1], inp.shape[1]) )
#         for k in range(inp.shape[0]):
#             if i == k: 
#                 continue
#             xik = inp[i] - inp[k]
#             pik = compute_pij(A, inp, i, k)
#             term = pik * np.outer(xik, xik)
#             #print 'term=',term
#             first_term += term
#             if label[k] == label[i]:
#                 second_term += term
#         first_term *= p
#         #print 'i,1st,2nd:',i, first_term, second_term
#         A += lr * (first_term - second_term)
#     return A

# def transform(A, inp):
#     out = np.zeros(inp.shape)
#     for i in range(len(out)):
#         out[i] = applyA(A, inp[i])
#     return out 


# import pdb

# import numpy as np

# from scipy.optimize import (
#     check_grad,
#     fmin_cg,
#     fmin_ncg,
#     fmin_bfgs,
# )

# from sklearn.base import (
#     BaseEstimator,
#     TransformerMixin,
# )

# from sklearn.preprocessing import (
#     StandardScaler,
# )


# def square_dist(x1, x2=None):
#     """If x1 is NxD and x2 is MxD (default x1), return NxM square distances."""

#     if x2 is None:
#         x2 = x1

#     return (
#         np.sum(x1 * x1, 1)[:, np.newaxis] +
#         np.sum(x2 * x2, 1)[np.newaxis, :] -
#         np.dot(x1, (2 * x2.T))
#     )


# def nca_cost(A, xx, yy, reg):
#     """Neighbourhood Components Analysis: cost function and gradients

#         ff, gg = nca_cost(A, xx, yy)

#     Evaluate a linear projection from a D-dim space to a K-dim space (K<=D).
#     See Goldberger et al. (2004).

#     Inputs:
#         A  KxD Current linear transformation.
#         xx NxD Input data
#         yy Nx1 Corresponding labels, taken from any discrete set

#     Outputs:
#         ff 1x1 NCA cost function
#         gg KxD partial derivatives of ff wrt elements of A

#     Motivation: gradients in existing implementations, and as written in the
#     paper, have the wrong scaling with D. This implementation should scale
#     correctly for problems with many input dimensions.

#     Note: this function should be passed to a MINIMIZER.

#     """

#     N, D = xx.shape
#     assert(yy.size == N)
#     assert(A.shape[1] == D)
#     K = A.shape[0]

#     # Cost function:
#     zz = np.dot(A, xx.T)  # KxN

#     # TODO Subsample part of data to compute loss on.
#     # kk = np.exp(-square_dist(zz.T, zz.T[idxs]))  # Nxn
#     # kk[idxs, np.arange(len(idxs))] = 0

#     ss = square_dist(zz.T)
#     np.fill_diagonal(ss, np.inf)
#     mm = np.min(ss, axis=0)
#     kk = np.exp(mm - ss)  # NxN
#     np.fill_diagonal(kk, 0)
#     Z_p = np.sum(kk, 0)  # N,
#     p_mn = kk / Z_p[np.newaxis, :]  # P(z_m | z_n), NxN
#     mask = yy[:, np.newaxis] == yy[np.newaxis, :]
#     p_n = np.sum(p_mn * mask, 0)  # 1xN
#     ff = - np.sum(p_n)

#     # Back-propagate derivatives:
#     kk_bar = - (mask - p_n[np.newaxis, :]) / Z_p[np.newaxis, :]  # NxN
#     ee_bar = kk * kk_bar
#     zz_bar_part = ee_bar + ee_bar.T
#     zz_bar = 2 * (np.dot(zz, zz_bar_part) - (zz * np.sum(zz_bar_part, 0)))  # KxN
#     gg = np.dot(zz_bar, xx)  # O(DKN)

#     if reg > 0:
#         ff = ff + reg * np.dot(A.ravel(), A.ravel())
#         gg = gg + 2 * reg * A

#     return ff, gg


# def nca_cost_batch(self, A, xx, yy, idxs):

#     N, D = xx.shape
#     n = len(idxs)

#     assert(yy.size == N)
#     assert(A.shape[1] == D)

#     K = A.shape[0]

#     # Cost function:
#     zz = np.dot(A, xx.T)  # KxN
#     Z_p = np.sum(kk, 0)  # N,
#     p_mn = kk / Z_p[np.newaxis, :]  # P(z_m | z_n), NxN
#     mask = yy[:, np.newaxis] == yy[np.newaxis, :]
#     p_n = np.sum(p_mn * mask, 0)  # 1xN
#     ff = - np.sum(p_n)

#     # Back-propagate derivatives:
#     kk_bar = - (mask - p_n[np.newaxis, :]) / Z_p[np.newaxis, :]  # NxN
#     zz_bar_part = kk * (kk_bar + kk_bar.T)
#     zz_bar = 2 * (np.dot(zz, zz_bar_part) - (zz * sum(zz_bar_part, 0)))  # KxN
#     gg = np.dot(zz_bar, xx)  # O(DKN)

#     return ff, gg


# class NCA(BaseEstimator, TransformerMixin):
#     def __init__(self, reg=0, dim=None, optimizer='cg'):
#         self.reg = reg
#         self.K = dim
#         self.standard_scaler = StandardScaler()

#         if optimizer in ('cg', 'conjugate_gradients'):
#             self._fit = self._fit_conjugate_gradients
#         elif optimizer in ('gd', 'gradient_descent'):
#             self._fit = self._fit_gradient_descent
#         elif optimizer in ('mb', 'mini_batches'):
#             self._fit = self._fit_mini_batches
#         else:
#             raise ValueError("Unknown optimizer {:s}".format(optimizer))

#     def fit(self, X, y):

#         N, D = X.shape

#         if self.K is None:
#             self.K = D

#         self.A = np.random.randn(self.K, D) / np.sqrt(N)

#         X = self.standard_scaler.fit_transform(X)
#         return self._fit(X, y)

#     def _fit_gradient_descent(self, X, y):
#         # Gradient descent.
#         self.learning_rate = 0.001
#         self.error_tol = 0.001
#         self.max_iter = 1000

#         curr_error = None

#         # print(check_grad(costf, costg, 0.1 * np.random.randn(self.K * D)))
#         # idxs = list(sorted(random.sample(range(len(X)), 100)))

#         for it in range(self.max_iter):

#             f, g = nca_cost(self.A, X, y, self.reg)
#             self.A -= self.learning_rate * g

#             prev_error = curr_error
#             curr_error = f

#             print('{:4d} {:+.6f}'.format(it, curr_error))

#             if prev_error and np.abs(curr_error - prev_error) < self.error_tol:
#                 break

#         return self

#     def _fit_conjugate_gradients(self, X, y):
#         N, D = X.shape

#         def costf(A):
#             f, _ = nca_cost(A.reshape([self.K, D]), X, y, self.reg)
#             return f 

#         def costg(A):
#             _, g = nca_cost(A.reshape([self.K, D]), X, y, self.reg)
#             return g.ravel()

#         # print(check_grad(costf, costg, 0.1 * np.random.randn(self.K * D)))
#         self.A = fmin_cg(costf, self.A.ravel(), costg, maxiter=400)
#         self.A = self.A.reshape([self.K, D])
#         return self

#     def fit_transform(self, X, y):
#         self.fit(X, y)
#         return self.transform(X)

#     def transform(self, X):
#         return np.dot(self.standard_scaler.transform(X), self.A.T)
