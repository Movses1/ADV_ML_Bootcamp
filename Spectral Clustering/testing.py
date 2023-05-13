import numpy as np
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from spectral import SpectralClustering
from sklearn.cluster import SpectralClustering as SC
from sklearn.cluster import KMeans

n_samples = 300
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]

X, y = make_blobs(n_samples=n_samples, random_state=170)
X = np.dot(X, transformation)

km = KMeans(3)
clusterer = SpectralClustering(3)
clusterer_skl = SC(3,
                   eigen_solver="arpack",
                   affinity="nearest_neighbors")

fig, axs = plt.subplots(1, 3)
axs[0].title.set_text('k-means')
axs[0].scatter(X[:, 0], X[:, 1], s=10, c=km.fit_predict(X))
axs[1].title.set_text('custom SC')
axs[1].scatter(X[:, 0], X[:, 1], s=10, c=clusterer.fit_predict(X))
axs[2].title.set_text('sklearn SC')
axs[2].scatter(X[:, 0], X[:, 1], s=10, c=clusterer_skl.fit_predict(X))

plt.show()
