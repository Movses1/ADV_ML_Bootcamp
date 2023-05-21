import numpy as np
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt


class T_SNE:
    def __init__(self, n_components=2, max_iter=300, l_r=0.01, acel=0.01):
        self.n_components = n_components
        self.max_iter = max_iter
        self.l_r = l_r
        self.datapoints = []
        self.D = 0
        self.D_sum = 0
        self.prev_state=[]
        self.acel=acel

    def _calculate_probabilities(self, X):
        D = distance_matrix(X, X)
        self.prob = (1 + D ** 2) ** (-1)
        return self.prob / self.prob.sum(axis=1)

    def _calculate_gradients(self, X):
        q = self._calculate_probabilities(X)
        grads = []
        for i in range(X.shape[0]):
            grad = 4 * np.sum((self.p[i] - q[i])[:, np.newaxis] * (X[i] - X), axis=0)
            # grad = 4 * np.sum((self.p[i]-q[i])[:,np.newaxis]*(X[i]-X)/self.prob[i][:,np.newaxis], axis=0)
            grads.append(grad)
        return np.array(grads)

    def fit_transform(self, X):
        self.p = self._calculate_probabilities(X)
        """for i in range(X.shape[0]):
            for j in range(i + 1, X.shape[0]):
                self.p[i, j] = (self.p[i, j] + self.p[j, i]) / (2 * X.shape[0])
                self.p[j, i] = self.p[i, j]"""

        self.df_shape = (X.shape[0], self.n_components)
        np.random.seed(1)
        self.datapoints = np.random.normal(0, 1, size=self.df_shape)
        self.prev_state = self.datapoints

        for _ in range(self.max_iter):
            gradients = self._calculate_gradients(self.datapoints)
            step = - gradients * self.l_r + self.acel * (self.datapoints - self.prev_state)
            self.prev_state = self.datapoints

            self.datapoints += step
            print(f'{_}/{self.max_iter}')

        return self.datapoints