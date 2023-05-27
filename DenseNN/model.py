import numpy as np
from layers import DenseLayer, Dropout, BatchNormalization


class Model:
    def __init__(self, layer_arr, loss='mse'):
        """
        loss: mse for regression, bce for classification
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
            prev_n = layer.neurons
        self.layers = layer_arr

    def predict(self, X):
        ans = X
        if len(X.shape) == 1:
            ans = ans.reshape(1, -1)

        for l in range(1, len(self.layers)):
            ans = self.layers[l]._feedforward(ans)
        return np.array(ans)

    def fit(self, X, Y, epochs=50, batch_size=32, lr=0.001):
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)
        scaling_factor = X.shape[0] // batch_size #num of batches

        for _ in range(epochs):
            print('epoch', _, end=' ')
            losses = 0

            indxs = np.arange(X.shape[0])
            np.random.shuffle(indxs)
            for j in range(0, X.shape[0], batch_size):
                indx = indxs[j:j + batch_size]
                ans = self.predict(X[indx])
                neuron_grads = 0
                if self.loss == 'mse':
                    neuron_grads = 2 * (ans - Y[indx]) / Y.shape[1]
                    losses += np.mean((ans - Y[indx]) ** 2 / Y.shape[1])
                elif self.loss == 'bce':
                    epsilon = 1e-7
                    neuron_grads = (-Y[indx] / (ans + epsilon) + (1 - Y[indx]) / (1 - ans + epsilon)) / 2
                    losses -= np.mean(Y[indx] * np.log(ans + epsilon) + (1 - Y[indx]) * np.log(1 - ans + epsilon))

                for i in range(len(self.layers) - 1, 0, -1):
                    neuron_grads = self.layers[i]._backpropagate(neuron_grads)
                    if type(self.layers[i]) not in [Dropout, BatchNormalization]:
                        self.layers[i]._apply_grads(lr / scaling_factor)
            print('loss', losses / scaling_factor)
