import numpy as np


class Adam:
    def __init__(self, beta_1=0.9, beta_2=0.999, shape=(1,)):
        self.beta1 = beta_1
        self.beta2 = beta_2
        self.beta1_deg = beta_1
        self.beta2_deg = beta_2
        self.v = np.zeros(shape)
        self.r = np.zeros(shape)

    def __call__(self, grad, lr=0.01):
        self.v = self.beta1 * self.v + (1 - self.beta1) * grad
        self.r = self.beta2 * self.r + (1 - self.beta2) * grad ** 2

        v_hat = self.v / (1 - self.beta1_deg)
        r_hat = self.r / (1 - self.beta2_deg)
        self.beta1_deg *= self.beta1
        self.beta2_deg *= self.beta2

        eps = 1e-7
        return lr * v_hat / (np.sqrt(r_hat) + eps)
