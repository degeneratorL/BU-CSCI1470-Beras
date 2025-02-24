from collections import defaultdict
import numpy as np

class BasicOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def apply_gradients(self, trainable_params, grads):
        #return NotImplementedError
        for param, grad in zip(trainable_params, grads):
            new_value = param - self.learning_rate * grad
            param.assign(new_value)


class RMSProp:
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v = defaultdict(lambda: 0)

    def apply_gradients(self, trainable_params, grads):
        #return NotImplementedError
        for param, grad in zip(trainable_params, grads):
            key = id(param)  # identify the parameter uniquely
            # update v
            self.v[key] = self.beta * self.v[key] + (1 - self.beta) * (grad ** 2)
            # update param
            new_value = param - (self.learning_rate / np.sqrt(self.v[key] + self.epsilon)) * grad
            param.assign(new_value)


class Adam:
    def __init__(
        self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
    ):


        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = defaultdict(lambda: 0)         # First moment zero vector
        self.v = defaultdict(lambda: 0)         # Second moment zero vector.
        self.t = 0                              # Time counter

        self.amsgrad = amsgrad

    def apply_gradients(self, trainable_params, grads):
        #return NotImplementedError
        self.t += 1  # increment time step
        for param, grad in zip(trainable_params, grads):
            key = id(param)
            
            # update first moment
            self.m[key] = self.beta_1 * self.m[key] + (1 - self.beta_1) * grad
            # update second moment
            self.v[key] = self.beta_2 * self.v[key] + (1 - self.beta_2) * (grad ** 2)

            # bias correction
            m_hat = self.m[key] / (1.0 - self.beta_1 ** self.t)
            v_hat = self.v[key] / (1.0 - self.beta_2 ** self.t)

            # update param
            new_value = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            param.assign(new_value)
