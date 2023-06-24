from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
from model import Model
from layers import DenseLayer, Dropout, InpLayer, Conv2D
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

data = load_digits(n_class=2)
x = data['data'].reshape(data['data'].shape[0], 8, 8, 1)
y = data['target']

model = Model([InpLayer(shape=np.array([8, 8, 1])),
               Conv2D(kernel_size=(4, 3), stride=(1, 1), filters=10, activation='relu'),
               Conv2D(kernel_size=(3, 4), stride=(1, 1), filters=5, activation='relu'),
               DenseLayer(neurons=1, activation='sigmoid')],  # classification
              loss='bce',
              )

model.fit(x, y, epochs=30, batch_size=32, lr=1e-2)

pred = model.predict(x)

roc = roc_curve(y, pred)
plt.plot(roc[0], roc[1], label='test')
plt.show()
