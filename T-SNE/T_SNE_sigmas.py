import numpy as np
from scipy.spatial import distance_matrix
from IPython.display import clear_output
import os

class T_SNE:
    def __init__(self, n_components=2, perplexity=30, max_iter=1000, l_r=10, acel=0.5):
        self.n_components = n_components
        self.perplexity = perplexity
        self.max_iter = max_iter
        self.l_r = l_r
        self.datapoints = []
        self.D = 0
        self.D_sum = 0
        self.p = []
        self.acel = acel
        self.sigmas = []

    def _perplexity(self, p):
        deg = -np.sum(p * np.log2(p))
        return 2 ** deg

    def _bin_search_sigma(self, X):
        self.sigmas = []
        D = distance_matrix(X, X) / 1

        for i in range(X.shape[0]):
            print(f'searching sigmas {i}/{X.shape[0]}')
            sigma_min = 0
            sigma_max = np.inf
            sigma = 4

            for _ in range(50):
                p = np.exp((-D[i] ** 2 / (2 * sigma ** 2)).astype('float64'))
                p[i] = 1e-8
                p = p / p.sum()
                perp = self._perplexity(p)

                if np.abs(perp - self.perplexity) < 1e-5:
                    break
                if perp < self.perplexity:
                    sigma_min = sigma
                    if sigma_max == np.inf:
                        sigma *= 2
                    else:
                        sigma = (sigma_min + sigma_max) / 2
                else:
                    sigma_max = sigma
                    sigma = (sigma_min + sigma_max) / 2

            self.sigmas.append(sigma)
        self.sigmas = np.array(self.sigmas)

        os.system('cls')
        return 0

    def _calculate_exp_probabilities(self, X):
        self.D = distance_matrix(X, X)
        prob = np.exp(-self.D ** 2 / (2 * self.sigmas[:, np.newaxis] ** 2))
        prob = prob / prob.sum(axis=0)
        return (prob + prob.T) / (2)  # *X.shape[0])

    def _calculate_exp_gradients(self, X):
        q = self._calculate_exp_probabilities(X)
        grads = []
        for i in range(X.shape[0]):
            # grad = 2 * np.sum(((self.p[i]-q[i]+self.p[:,i]-q[:,i])*self.D[i])[:,np.newaxis]*(X[i]-X), axis=0)
            grad = 4 * np.sum((self.p[i] - q[i])[:, np.newaxis] * (X[i] - X) / (self.D[i] + 1)[:, np.newaxis], axis=0)
            grads.append(grad)
        return np.array(grads)

    def fit_transform(self, X, do_exp=False, plot_learning=False, mnist_colors=False):
        self._bin_search_sigma(X)

        self.p = self._calculate_exp_probabilities(X)

        self.df_shape = (X.shape[0], self.n_components)
        np.random.seed(1)
        self.datapoints = np.random.normal(0, 1, size=self.df_shape)
        self.prev_state = self.datapoints

        for _ in range(self.max_iter):
            gradients = self._calculate_exp_gradients(self.datapoints)
            step = -gradients * self.l_r + self.acel * (self.datapoints - self.prev_state)
            self.prev_state = self.datapoints

            self.datapoints = self.datapoints + step
            print(f'gradient descent {_}/{self.max_iter}')

        return self.datapoints