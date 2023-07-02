import numpy as np
import tensorflow as tf
from optimizers import Adam


class Dropout:
    def __init__(self, rate=0.1, neurons=32):
        self.rate = 0.1
        self.scale = 1 / (1 - rate)
        self.neurons = neurons
        self.weights = 0

    def _init_weights(self):
        self.weights = np.random.rand(self.neurons) > self.rate
        self.weights = self.weights * self.scale

    def _feedforward(self, X, fitting=True):
        if fitting:
            self._init_weights()
            return X * self.weights
        return X

    def _backpropagate(self, neuron_grads):
        return neuron_grads  # * self.scale**(-1)


class DenseLayer:
    def __init__(self, activation='linear', neurons=64, include_bias=True):
        """
        activation can be: 'relu', 'sigmoid' everything else is treated as 'linear'
        """
        self.neurons = neurons
        self.include_bias = include_bias
        self.activation = activation
        self.weights = []

    @staticmethod
    def relu(X):
        return tf.cast((X > 0), tf.float64) * X

    @staticmethod
    def sigmoid(X):
        return 1 / (1 + tf.math.exp(-X))

    def apply_activation(self, X):
        if self.activation == 'relu':
            return DenseLayer.relu(X)
        elif self.activation == 'sigmoid':
            return DenseLayer.sigmoid(X)
        return X

    def _init_weights(self, inp_size, layer_indx, indx):
        """
        Xavier Glorot initialization
        """
        std = inp_size ** (-layer_indx / 2)
        self.w_adm = Adam(shape=(inp_size, self.neurons))
        self.weights = np.random.normal(0, std, size=(inp_size, self.neurons))
        self.weights = tf.Variable(self.weights,
                                   dtype=tf.float64,
                                   name=f'W{indx}',
                                   trainable=True)
        if self.include_bias:
            self.b_adm = Adam(shape=(self.neurons,))
            self.bias = tf.Variable(tf.zeros(shape=(self.neurons,), dtype=tf.float64),
                                    name=f'B{indx}',
                                    trainable=True)
            return self.weights, self.bias
        return self.weights

    def _feedforward(self, inp):
        """
        inp is a 1D array representing the previous layer
        """
        ans = tf.Variable(0, dtype=tf.float64)

        if self.include_bias:
            temp_var = inp @ self.weights
            ans = temp_var + self.bias
        else:
            ans = inp @ self.weights
        ans = self.apply_activation(ans)

        return ans

    def _apply_grads(self, lr, grads):
        if self.include_bias:
            self.weights.assign_sub(self.w_adm(grads[0], lr))
            self.bias.assign_sub(self.b_adm(grads[1], lr))
        else:
            self.weights.assign_sub(self.w_adm(grads, lr))
