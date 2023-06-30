import numpy as np
from optimizers import Adam

#np.random.seed(2)
epsilon=1e-7

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
    sgm = sigmoid(X)
    return sgm * (1 - sgm)


def softmax(X):
    exps = np.exp(X)
    sm = exps.sum(axis=tuple(np.arange(1, len(X.shape))))
    #print(sm.reshape(sm.shape+tuple(1 for _ in exps.shape[1:])))
    return exps / (sm.reshape(sm.shape+tuple(1 for _ in exps.shape[1:]))+epsilon)


def softmax_derivative(X):
    sftmx = softmax(X)
    return sftmx * (1 - sftmx)


def tanh(X):
    return 1 - (2 * np.exp(-X) / (np.exp(X) + np.exp(-X)))


def tanh_derivative(X):
    return 1 - tanh(X) ** 2


def apply_activation(X, activation):
    if activation == 'relu':
        return relu(X)
    elif activation == 'sigmoid':
        return sigmoid(X)
    elif activation == 'tanh':
        return tanh(X)
    elif activation == 'softmax':
        return softmax(X)
    return X


def apply_derivative(X, activation):
    if activation == 'relu':
        return relu_derivative(X)
    elif activation == 'sigmoid':
        return sigmoid_derivative(X)
    elif activation == 'tanh':
        return tanh_derivative(X)
    elif activation == 'softmax':
        return softmax_derivative(X)
    return 1


class InpLayer:
    def __init__(self, shape):
        self.neurons = shape

    def _feedforward(self, inp):
        return inp


class RNN:
    def __init__(self, activation='tanh', neurons=20, include_bias=True):
        self.activation = activation
        self.neurons = neurons
        self.include_bias = include_bias
        self.weights_inp = []
        self.weights_h = []

        self.grads_inp = []
        self.grads_h = []
        self.grads_bias = []
        self.next_timestep_grad = []

        self.inp_history = []
        self.ans_history = []

    def _reset_grads(self):
        self.next_timestep_grad = np.zeros(self.neurons)
        self.grads_inp = np.zeros(self.weights_inp.shape)
        self.grads_h = np.zeros(self.weights_h.shape)
        if self.include_bias:
            self.grads_bias = np.zeros(self.neurons)

    def _init_weights(self, inp_size, layer_indx):
        if type(inp_size) is np.ndarray:
            inp_size = inp_size.prod()

        std_inp = np.sqrt(1 / (inp_size + self.neurons))
        std_h = np.sqrt(0.5 / self.neurons)
        self.weights_inp = np.random.uniform(-std_inp, std_inp, size=(inp_size, self.neurons))
        self.weights_h = np.random.uniform(-std_h, std_h, size=(self.neurons, self.neurons))
        if self.include_bias:
            self.bias = np.zeros(self.neurons)

        self._reset_grads()

    def _feedforward(self, inp):
        """
        inp is a 1D array representing the previous layer
        """
        if len(inp.shape) > 2:
            inp = inp.reshape(inp.shape[0], -1)

        ans = self.ans_history[-1] @ self.weights_h
        ans += inp @ self.weights_inp
        if self.include_bias:
            ans += self.bias

        self.inp_history.append(inp)
        self.ans_history.append(ans)

        return apply_activation(ans, self.activation)

    def _backpropagate(self, neuron_grads):
        neuron_grads *= apply_derivative(self.ans_history[-1], self.activation)
        neuron_grads += self.next_timestep_grad
        # saving the gradient coming from hidden state
        self.next_timestep_grad = neuron_grads @ self.weights_h.T

        g = self.inp_history[-1].reshape(self.inp_history[-1].shape + (1,)) @ \
            neuron_grads.reshape(neuron_grads.shape[0], 1, -1)
        self.grads_inp += g.mean(axis=0)

        g = self.inp_history[-1].reshape(self.inp_history[-1].shape + (1,)) @ \
            neuron_grads.reshape(neuron_grads.shape[0], 1, -1)
        self.grads_h += g.mean(axis=0)

        if self.include_bias:
            self.grads_bias += neuron_grads.mean(axis=0)

        self.ans_history.pop()
        self.inp_history.pop()

        return neuron_grads @ self.weights_inp.T

    def _apply_grads(self, lr):
        self.weights_h -= lr * self.grads_h
        self.weights_inp -= lr * self.grads_inp
        if self.include_bias:
            self.bias -= lr * self.grads_bias
        self._reset_grads()


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

        # std = inp_size.prod() ** (-layer_indx / 2)                  # xavier Glorot
        # std = np.sqrt(2 / (inp_size.prod() + self.neurons.prod()))  # Glorot uniform
        std = np.sqrt(1/inp_size.prod())                            # He normal

        # weight dimensions is (h, w, d, filters)
        # self.weights = np.random.normal(0, std, size=(self.kernel_size + (inp_size[2], self.filters)))
        self.weights = np.random.uniform(-std, std, size=(self.kernel_size + (inp_size[2], self.filters)))
        self.w_adm = Adam(shape=self.weights.shape)

        # print(self.weights.shape)
        if self.include_bias:
            self.bias = np.zeros(self.filters)
            self.b_adm = Adam(shape=(self.bias.shape))

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

        return apply_activation(self.ans, self.activation)

    def _backpropagate(self, neuron_grads):
        """
         grads shape should be (batch, h, w, f)
         return shape (batch, h, w, f)
        """
        if len(neuron_grads.shape) == 2:
            neuron_grads = neuron_grads.reshape(
                (neuron_grads.shape[0], len(self.height), len(self.width), self.filters))
        neuron_grads *= apply_derivative(self.ans, self.activation)

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
        self.weights -= self.w_adm(self.weight_grads, lr)
        self.bias -= self.b_adm(self.bias_grads, lr)
        #self.weights -= lr * self.weight_grads
        #self.bias -= lr * self.bias_grads


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
        if type(inp_size) is np.ndarray:
            inp_size = inp_size.prod()

        # std = inp_size ** (-layer_indx / 2)         # xavier glorot
        std = np.sqrt(1 / (inp_size + self.neurons))  # glorot uniform

        # self.weights = np.random.normal(0, std, size=(inp_size, self.neurons))
        self.weights = np.random.uniform(-std, std, size=(inp_size, self.neurons))

        if self.include_bias:
            self.weights = np.append(self.weights, np.zeros((1, self.neurons)), axis=0)

        self.w_adm = Adam(shape=self.weights.shape)

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

        return apply_activation(self.ans, self.activation)

    def _backpropagate(self, neuron_grads):
        neuron_grads *= apply_derivative(self.ans, self.activation)

        self.grads = self.inp.reshape(self.inp.shape + (1,)) @ neuron_grads.reshape(neuron_grads.shape[0], 1, -1)
        self.grads = self.grads.mean(axis=0)

        if self.include_bias:
            return neuron_grads @ self.weights[:-1].T
        return neuron_grads @ self.weights.T

    def _apply_grads(self, lr):
        self.weights -= self.w_adm(self.grads, lr)
        # self.weights -= lr * self.grads
