import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian


class SpectralClustering:
    def __init__(self, n_clusters=1, seed=1):
        self.eigenvecs = []
        self.normalized_embedding = []
        self.n_clusters = n_clusters
        self.seed = seed

    def fit(self, X, k_neighbors=10):
        # K_neighbors=10 to match sklearn

        adj_matrix = kneighbors_graph(X, k_neighbors, mode='connectivity')
        adj_matrix = (adj_matrix + adj_matrix.T) / 2
        L = laplacian(adj_matrix, normed=True)  # L=D-A but using built-in function performs much faster

        _, self.eigenvecs = eigsh(L, k=self.n_clusters + 1, which='SM')
        # self.eigenvecs = self.eigenvecs[:, 1:]

        norms = np.linalg.norm(self.eigenvecs, axis=1)[:, np.newaxis]
        norms[norms == 0] = 1
        self.normalized_embedding = self.eigenvecs / norms
        return self.normalized_embedding

    def predict(self):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.seed)
        return kmeans.fit_predict(self.normalized_embedding)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict()
