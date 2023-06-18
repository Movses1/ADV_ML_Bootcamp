import numpy as np
import tensorflow as tf
from layers import Dropout, DenseLayer


class Model:
    def __init__(self, layer_arr, loss='mse'):
        """
        loss: mse for regrssion, bce for classification
        """
        self.loss = loss
        prev_n = 0
        norm_cnt = 0
        self.trainable_variables = []
        for ind, layer in enumerate(layer_arr):
            if ind != 0:
                if type(layer) is Dropout:
                    layer.neurons = prev_n
                    norm_cnt += 1
                else:
                    w = layer._init_weights(prev_n, ind - norm_cnt, ind)
                    if layer.include_bias:
                        self.trainable_variables.append(w[0])
                        self.trainable_variables.append(w[1])
                    else:
                        self.trainable_variables.append(w)
            prev_n = layer.neurons
        self.layers = layer_arr

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X = tf.Variable(X, dtype=tf.float64, name='features')
        ans = X

        for l in range(1, len(self.layers)):
            ans = self.layers[l]._feedforward(ans)

        return ans

    def fit(self, X, Y, epochs=50, batch_size=32, lr=0.001):
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)

        for _ in range(epochs):
            print('epoch', _, end=' ')
            losses = 0

            indxs = np.arange(X.shape[0])
            np.random.shuffle(indxs)
            for j in range(0, X.shape[0], batch_size):
                indx = indxs[j:j + batch_size]
                with tf.GradientTape() as tape:
                    ans = self.predict(X[indx])
                    loss_val = 0
                    if self.loss == 'mse':
                        loss_val = tf.reduce_mean((ans - Y[indx]) ** 2)
                    elif self.loss == 'bce':
                        epsilon = 1e-7
                        loss_val -= tf.reduce_mean(
                            Y[indx] * tf.math.log(ans + epsilon) + (1 - Y[indx]) * tf.math.log(1 - ans + epsilon)) / 2
                    losses += loss_val

                grads = tape.gradient(loss_val, self.trainable_variables)
                grad_indx = 0
                for l in range(1, len(self.layers)):
                    if type(self.layers[l]) is Dropout:
                        continue
                    if self.layers[l].include_bias:
                        self.layers[l]._apply_grads(lr, grads[grad_indx:grad_indx + 2])
                        grad_indx += 2
                    else:
                        self.layers[l]._apply_grads(lr, grads[grad_indx])
                        grad_indx += 1
            print('loss -', float(losses) / (X.shape[0] // batch_size))
