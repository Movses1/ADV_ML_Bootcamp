from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
from model import Model
from layers import DenseLayer, Dropout, InpLayer, Conv2D
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
from matplotlib import cm

class_cnt = 10
data = load_digits(n_class=class_cnt)
x = data['data'].reshape(data['data'].shape[0], 8, 8, 1)
y1 = data['target']
y = np.zeros((y1.size, class_cnt))
y[np.arange(y1.size), y1] = 1

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

model = Model([InpLayer(shape=np.array([8, 8, 1])),
               Conv2D(kernel_size=(3, 3), stride=(1, 1), filters=10, activation='relu'),
               Conv2D(kernel_size=(3, 3), stride=(1, 1), filters=5, activation='relu'),
               DenseLayer(neurons=100, activation='relu'),
               DenseLayer(neurons=class_cnt, activation='softmax')],  # classification
              loss='cce',
              )

model.fit(X_train, y_train, epochs=30, batch_size=32, lr=1e-3)

preds = model.predict(X_test)
preds1 = model.predict(X_train)
for i in range(class_cnt):
    roc = roc_curve(y_test[:, i], preds[:, i])
    plt.plot(roc[0], roc[1], label=f'num {i} val', c=cm.hot(i / class_cnt))   # validation
    roc = roc_curve(y_train[:, i], preds1[:, i])
    plt.plot(roc[0], roc[1], linestyle='--', c=cm.hot(i / class_cnt))         # train
plt.legend()
plt.show()
