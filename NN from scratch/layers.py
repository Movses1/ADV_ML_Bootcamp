import numpy as np


np.random.seed(1)


class Dropout:
    def __init__(self, rate=0.1, neurons=32):
        self.rate = rate
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


class BatchNormalization:
    def __init__(self, neurons=32):
        self.neurons = 32
        self.min = 0
        self.max = 0

    def _feedforward(self, X):
        self.min = np.min(X, axis=0)
        X -= self.min
        self.max = np.max(X, axis=0)
        X /= (self.max + 1e-9)
        return X

    def _backpropagate(self, neuron_grads):
        return neuron_grads * self.max


def relu(X):
    return (X > 0) * X


def relu_derivative(X):
    return X > 0


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def sigmoid_derivative(X):
    return sigmoid(X) * (1 - sigmoid(X))


class InpLayer:
    def __init__(self, shape):
        self.neurons = shape

    def _feedforward(self, inp):
        return inp


class Conv2D:
    def __init__(self, activation='linear', kernel_size=(3, 3), stride=(1, 1), filters=10, include_bias=True):
        self.activation = activation
        self.kernel_size = kernel_size
        self.stride = stride
        self.filters = filters
        self.include_bias = include_bias
        self.weights = []
        self.neurons = []  # represents the output size

    def _init_weights(self, inp_size, layer_indx):
        self.height = np.arange(0, inp_size[0] - self.kernel_size[0] + 1, self.stride[0])
        self.width = np.arange(0, inp_size[1] - self.kernel_size[1] + 1, self.stride[1])
        self.neurons = np.array([len(self.height), len(self.width), self.filters])  # output size

        std = inp_size.prod() ** (-layer_indx / 2)

        # weight dimensions is (h, w, d, filters)
        self.weights = np.random.normal(0, std, size=(self.kernel_size + (inp_size[2], self.filters)))
        # print(self.weights.shape)
        if self.include_bias:
            self.bias = np.zeros(self.filters)

    def _feedforward(self, inp):
        """
        inp has shape (batch, h, w, d)
        out has shape (batch, h, w, f)
        """
        self.inp = inp[:, :, :, :, np.newaxis]
        self.ans = np.zeros((inp.shape[0], len(self.height), len(self.width), self.filters))
        for h in self.height:
            for w in self.width:
                inp1 = self.inp[:, h:h + self.kernel_size[0], w:w + self.kernel_size[1], :, :]
                ans1 = (inp1 * self.weights + self.bias).sum(axis=(1, 2, 3))
                self.ans[:, h, w, :] = ans1

        if self.activation == 'relu':
            return relu(self.ans)
        elif self.activation == 'sigmoid':
            return sigmoid(self.ans)

        return self.ans

    def _backpropagate(self, neuron_grads):
        """
         grads shape should be (batch, h, w, f)
         return shape (batch, h, w, f)
        """
        if len(neuron_grads.shape) == 2:
            neuron_grads = neuron_grads.reshape(
                (neuron_grads.shape[0], len(self.height), len(self.width), self.filters))
        if self.activation == 'relu':
            neuron_grads *= relu_derivative(self.ans)
        elif self.activation == 'sigmoid':
            neuron_grads *= sigmoid_derivative(self.ans)

        inp_grads = np.zeros(self.inp.shape[:-1])
        self.weight_grads = np.zeros(self.weights.shape)
        self.bias_grads = neuron_grads.sum(axis=(0, 1, 2))

        for h_o, h in enumerate(self.height):
            for w_o, w in enumerate(self.width):
                self.weight_grads += (
                        neuron_grads[:, h_o:h_o + 1, w_o:w_o + 1, np.newaxis, :] * self.inp[:,
                                                                                   h:h + self.kernel_size[0],
                                                                                   w:w + self.kernel_size[1], :]).mean(
                    axis=0)
                inp_grads[:, h:h + self.kernel_size[0], w:w + self.kernel_size[1], :] += (
                        neuron_grads[:, h_o:h_o + 1, w_o:w_o + 1, np.newaxis, :] * self.weights).sum(axis=-1)

        return inp_grads

    def _apply_grads(self, lr):
        self.weights -= lr * self.weight_grads
        self.bias -= lr * self.bias_grads


class DenseLayer:
    def __init__(self, activation='linear', neurons=64, include_bias=True):
        """
        activation can be: 'relu', 'sigmoid' everything else is treated as 'linear'
        """
        self.neurons = neurons
        self.include_bias = include_bias
        self.activation = activation
        self.weights = []

    def _init_weights(self, inp_size, layer_indx):
        """
        Xavier Glorot initialization
        """
        if type(inp_size) is np.ndarray:
            inp_size = inp_size.prod()

        std = inp_size ** (-layer_indx / 2)
        self.weights = np.random.normal(0, std, size=(inp_size, self.neurons))
        if self.include_bias:
            self.weights = np.append(self.weights, np.zeros((1, self.neurons)), axis=0)

    def _feedforward(self, inp):
        """
        inp is a 1D array representing the previous layer
        """
        if len(inp.shape) > 2:
            inp = inp.reshape(inp.shape[0], -1)
        if self.include_bias:
            inp = np.insert(inp, -1, 1, axis=-1)

        ans = inp @ self.weights
        self.inp = inp
        self.ans = ans

        if self.activation == 'relu':
            return relu(ans)
        elif self.activation == 'sigmoid':
            return sigmoid(ans)
        return ans

    def _backpropagate(self, neuron_grads):
        if self.activation == 'relu':
            neuron_grads *= relu_derivative(self.ans)
        elif self.activation == 'sigmoid':
            neuron_grads *= sigmoid_derivative(self.ans)

        self.grads = self.inp.reshape(self.inp.shape + (1,)) @ neuron_grads.reshape(neuron_grads.shape[0], 1, -1)
        self.grads = self.grads.mean(axis=0)

        if self.include_bias:
            return neuron_grads @ self.weights[:-1].T
        return neuron_grads @ self.weights.T

    def _apply_grads(self, lr):
        self.weights -= lr * self.grads
