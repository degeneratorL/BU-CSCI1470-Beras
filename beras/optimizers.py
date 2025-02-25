from collections import defaultdict
import numpy as np

class BasicOptimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def apply_gradients(self, trainable_params, grads):
        for param, grad in zip(trainable_params, grads):
            new_value = param - self.learning_rate * grad
            param.assign(new_value)


class RMSProp:
    """
    修正：将 mean_square -> v, 并保证在第一次见到 param 时初始化为 zeros_like(param).
    默认超参:
        learning_rate=0.01
        beta=0.9
        epsilon=1e-7
    """
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v = defaultdict(lambda: None)  # 存储二阶梯度的指数平滑

    def apply_gradients(self, weights, grads):
        for i, (weight, grad) in enumerate(zip(weights, grads)):
            self.v[i] = self.beta * self.v[i] + (1-self.beta) * (grad) ** 2
            weights[i] -= ((self.learning_rate) / ((self.v[i]) ** (1/2) + self.epsilon) * grad)


class Adam:
    """
    修正: 
    1) 避免过大的 learning_rate(原本0.3比较罕见, 
       如测试期望更小LR可改 0.001~0.01).
    2) 第一次见到 param 时, 初始化 m, v.
    默认超参:
        learning_rate=0.001
        beta_1=0.9, beta_2=0.999
        epsilon=1e-7
    """
    def __init__(
        self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
    ):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

        self.m = defaultdict(lambda: None)  # 一阶动量
        self.v = defaultdict(lambda: None)  # 二阶动量
        self.m_hat = defaultdict(lambda: 0)     # Expected value of first moment vector
        self.v_hat = defaultdict(lambda: 0)     # Expected value of second moment vector
        self.t = 0                           # 时间步

    def apply_gradients(self, weights, grads):
        self.t += 1
        # m represents momentum to propel gradient, v is like acceleration, adjusts learning rate
        for i in range(len(weights)):
            self.m[i] = (self.beta_1 * self.m[i]) + ((1-self.beta_1) * grads[i])
            self.v[i] = (self.beta_2 * self.v[i]) + ((1-self.beta_2) * (grads[i]**2))

            self.m_hat[i] = self.m[i] / (1-(self.beta_1 ** self.t))
            self.v_hat[i] = self.v[i] / (1-(self.beta_2 ** self.t))

            weights[i] -= self.learning_rate * (self.m_hat[i] / ((self.v_hat[i]) ** (1/2) + self.epsilon))