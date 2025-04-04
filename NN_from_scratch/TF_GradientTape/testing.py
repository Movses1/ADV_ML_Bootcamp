from sklearn.datasets import load_diabetes
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from model import Model
from layers import DenseLayer, Dropout
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

reg = True       # control regression or classification problem
last_layer = 0
loss = 0
if reg:
    last_layer = DenseLayer(neurons=1)
    our_loss = 'mse'
    data = load_diabetes()
else:
    last_layer = DenseLayer(neurons=1, activation='sigmoid')
    our_loss = 'bce'
    data = load_breast_cancer()

x = data['data']
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

model = Model([DenseLayer(neurons=x.shape[1], include_bias=False),
               DenseLayer(neurons=64, activation='relu'),
               Dropout(0.2),
               DenseLayer(neurons=64, activation='relu'),
               DenseLayer(neurons=64, activation='relu'),
               last_layer],  # classification
              loss=our_loss,
              )
model.fit(X_train, y_train, epochs=40, batch_size=32, lr=0.001)
preds = model.predict(X_test)
preds1 = model.predict(X_train)

if reg:
    print('train mse =', np.mean(((preds1 - y_train) ** 2)), 'mae =', np.mean(np.abs(preds1 - y_train)))
    print('test mse =', np.mean(((preds - y_test) ** 2)), 'mae =', np.mean(np.abs(preds - y_test)))

else:
    roc = roc_curve(y_test, preds)
    plt.plot(roc[0], roc[1], label='test')
    roc = roc_curve(y_train, preds1)
    plt.plot(roc[0], roc[1], label='train')
    plt.legend()
    plt.show()
