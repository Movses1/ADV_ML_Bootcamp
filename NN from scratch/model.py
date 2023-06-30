import numpy as np
from layers import DenseLayer, Dropout, BatchNormalization

#np.random.seed(2)

def calculate_loss(y_true, y_pred, loss='mse'):
    """
    :return: (gradients, loss)
    """
    n_g = 0
    l = 0
    epsilon = 1e-8
    y_pred += epsilon

    if loss == 'mse':
        n_g = 2 * (y_pred - y_true) / y_true.shape[1]
        l = np.mean((y_pred - y_true) ** 2)
    elif loss == 'bce':
        n_g = (-y_true / (y_pred + epsilon) + (1 - y_true) / (1 - y_pred + epsilon)) / 2
        l = -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon)) / 2
    elif loss == 'cce':
        """div = y_true / (y_pred + epsilon)
        sm = div.sum(axis=tuple(np.arange(1, len(div.shape))))
        sm = sm.reshape(sm.shape + tuple(1 for _ in div.shape[1:]))
        n_g = -div + (sm - div)
        l = -np.sum(y_true * np.log(y_pred))"""
        n_g = y_pred - y_true
        l = -np.sum(y_true * np.log(y_pred))

    return n_g, l


class Model:
    def __init__(self, layer_arr, loss='mse'):
        """
        loss: mse for regression, bce and cce for binary/categorical cross-enropy
        """
        self.loss = loss
        prev_n = 0
        norm_cnt = 0
        for ind, layer in enumerate(layer_arr):
            if ind != 0:
                if type(layer) in [Dropout, BatchNormalization]:
                    layer.neurons = prev_n
                    norm_cnt += 1
                else:
                    layer._init_weights(prev_n, ind - norm_cnt)
                # print(layer.weights.shape)
            prev_n = layer.neurons
        self.layers = layer_arr

    def predict(self, X, fitting=False):
        ans = X
        if len(X.shape) == 1:
            ans = ans.reshape(1, -1)

        for l in self.layers[1:]:
            if type(l) is Dropout and not fitting:
                continue
            else:
                ans = l._feedforward(ans)
        return np.array(ans)

    def fit(self, X, Y, epochs=50, batch_size=32, lr=0.001):
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)
        scaling_factor = X.shape[0] / batch_size  # num of batches

        for _ in range(epochs):
            print('epoch', _, end=' ')
            losses = 0

            indxs = np.arange(X.shape[0])
            np.random.shuffle(indxs)
            for j in range(0, X.shape[0], batch_size):
                indx = indxs[j:j + batch_size]
                ans = self.predict(X[indx], fitting=True)

                n_g, l = calculate_loss(Y[indx], ans, self.loss)
                neuron_grads = n_g
                losses += l

                for i in range(len(self.layers) - 1, 0, -1):
                    neuron_grads = self.layers[i]._backpropagate(neuron_grads)
                    if type(self.layers[i]) not in [Dropout, BatchNormalization]:
                        self.layers[i]._apply_grads(lr)  # / scaling_factor)
            print('loss', losses / scaling_factor)
