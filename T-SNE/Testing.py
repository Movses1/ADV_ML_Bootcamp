import numpy as np
from sklearn import datasets
from T_SNE_sigmas import T_SNE
from matplotlib import pyplot as plt

digits = datasets.load_digits()
digit_cnt=1700

imgs = digits['images'][:digit_cnt]
imgs = imgs.reshape((imgs.shape[0], -1))

t_sne = T_SNE(n_components=2, max_iter=200, l_r=5, acel=0.8)
x = t_sne.fit_transform(imgs)

plt.scatter(x[:, 0], x[:, 1], s=20, c=digits['target'][:digit_cnt])
plt.show()