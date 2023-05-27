from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
from model import Model
from layers import DenseLayer

data = load_diabetes()
x = data['data']
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

model = Model([DenseLayer(neurons=x.shape[1], include_bias=False),
               DenseLayer(neurons=64, activation='relu'),
               DenseLayer(neurons=64, activation='relu'),
               DenseLayer(neurons=1)],
              )
model.fit(X_train, y_train, epochs=50, batch_size=248, lr=0.001)
preds = model.predict(X_test)
print('mse =', np.mean(((preds - y_test) ** 2)), 'mae =', np.mean(np.abs(preds - y_test)))
