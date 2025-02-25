from collections import defaultdict
import numpy as np

"""
TODO: Implement all the apply_gradients for the 3 optimizers:
    - BasicOptimizer
    - RMSProp
    - Adam
"""

class BasicOptimizer:
    """
    This class represents a basic optimizer which simply applies the scaled gradients to the weights.

    TODO: Roadmap 5.
        - apply_gradients 
    """
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def apply_gradients(self, weights, grads):
        """
        given weights and grads, scale and then apply the gradients to (only trainable) weights
        You can assume that grads[i] is the gradient for weights[i]

        weights: the weights in the model we are training
        grads: the gradients ot those weights
        return: None
        """
        for i in range(len(weights)):
            if not weights[i].trainable: continue
            weights[i] -= self.learning_rate * grads[i]


class RMSProp:
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v = defaultdict(lambda: 0)

    def apply_gradients(self, weights, grads):
        ## Implement RMSProp optimization
        for i, (weight, grad) in enumerate(zip(weights, grads)):
            self.v[i] = self.beta * self.v[i] + (1-self.beta) * (grad) ** 2
            weights[i] -= ((self.learning_rate) / ((self.v[i]) ** (1/2) + self.epsilon) * grad)


class Adam:
    def __init__(
        self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
    ):
        self.amsgrad = amsgrad

        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = defaultdict(lambda: 0)         # First moment zero vector
        self.v = defaultdict(lambda: 0)         # Second moment zero vector.
        self.m_hat = defaultdict(lambda: 0)     # Expected value of first moment vector
        self.v_hat = defaultdict(lambda: 0)     # Expected value of second moment vector
        self.t = 0                              # Time counter

    def apply_gradients(self, weights, grads):
        self.t += 1
        # m represents momentum to propel gradient, v is like acceleration, adjusts learning rate
        for i in range(len(weights)):
            self.m[i] = (self.beta_1 * self.m[i]) + ((1-self.beta_1) * grads[i])
            self.v[i] = (self.beta_2 * self.v[i]) + ((1-self.beta_2) * (grads[i]**2))

            self.m_hat[i] = self.m[i] / (1-(self.beta_1 ** self.t))
            self.v_hat[i] = self.v[i] / (1-(self.beta_2 ** self.t))

            weights[i] -= self.learning_rate * (self.m_hat[i] / ((self.v_hat[i]) ** (1/2) + self.epsilon))
