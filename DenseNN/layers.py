import numpy as np


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
        return (X > 0) * X

    @staticmethod
    def relu_derivative(X):
        return X > 0

    @staticmethod
    def sigmoid(X):
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def sigmoid_derivative(X):
        return DenseLayer.sigmoid(X) * (1 - DenseLayer.sigmoid(X))

    def apply_activation(self, X):
        if self.activation == 'relu':
            return DenseLayer.relu(X)
        elif self.activation == 'sigmoid':
            return DenseLayer.sigmoid(X)
        return X

    def _init_weights(self, inp_size, layer_indx):
        """
        Xavier Glorot initialization
        """
        std = inp_size ** (-layer_indx / 2)
        self.weights = np.random.normal(0, std, size=(inp_size, self.neurons))
        if self.include_bias:
            self.weights = np.append(self.weights, np.zeros((1, self.neurons)), axis=0)

    def _feedforward(self, inp):
        """
        inp is a 1D array representing the previous layer
        """
        if self.include_bias:
            inp = np.insert(inp, -1, 1, axis=-1)

        ans = inp @ self.weights
        self.inp = inp
        self.ans = ans
        ans = self.apply_activation(ans)

        return ans

    def _backpropagate(self, neuron_grads):
        if self.activation == 'relu':
            neuron_grads *= DenseLayer.relu_derivative(self.ans)
        elif self.activation == 'sigmoid':
            neuron_grads *= DenseLayer.sigmoid_derivative(self.ans)

        self.grads = self.inp.reshape(self.inp.shape + (1,)) @ neuron_grads.reshape(neuron_grads.shape[0], 1, -1)
        self.grads = self.grads.mean(axis=0)

        if self.include_bias:
            return neuron_grads @ self.weights[:-1].T
        return neuron_grads @ self.weights.T

    def _apply_grads(self, lr):
        # print(self.weights.shape, self.grads.shape)
        self.weights -= lr * self.grads
