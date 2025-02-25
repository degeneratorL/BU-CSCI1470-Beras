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
    def __init__(self, learning_rate=0.01, beta=0.9, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v = defaultdict(lambda: None)  # 存储二阶梯度的指数平滑

    def apply_gradients(self, trainable_params, grads):
        for param, grad in zip(trainable_params, grads):
            key = id(param)
            # 第一次遇到 param, 初始化形状
            if self.v[key] is None:
                self.v[key] = np.zeros_like(param)

            # v[key] = beta * v[key] + (1-beta)*grad^2
            self.v[key] = self.beta * self.v[key] + (1.0 - self.beta) * (grad ** 2)

            # param = param - lr*(grad / sqrt(v+eps))
            new_value = param - self.learning_rate * grad / (np.sqrt(self.v[key]) + self.epsilon)
            param.assign(new_value)


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
        self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
    ):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

        self.m = defaultdict(lambda: None)  # 一阶动量
        self.v = defaultdict(lambda: None)  # 二阶动量
        self.t = 0                           # 时间步

    def apply_gradients(self, trainable_params, grads):
        self.t += 1
        for param, grad in zip(trainable_params, grads):
            key = id(param)
            # 若是第一次见到 param => 初始化
            if self.m[key] is None:
                self.m[key] = np.zeros_like(param)
                self.v[key] = np.zeros_like(param)

            # m = beta1*m + (1-beta1)*grad
            self.m[key] = self.beta_1 * self.m[key] + (1.0 - self.beta_1)*grad
            # v = beta2*v + (1-beta2)*(grad^2)
            self.v[key] = self.beta_2 * self.v[key] + (1.0 - self.beta_2)*(grad**2)

            # 偏差修正
            m_hat = self.m[key] / (1.0 - self.beta_1**self.t)
            v_hat = self.v[key] / (1.0 - self.beta_2**self.t)

            # param = param - lr*m_hat/(sqrt(v_hat)+eps)
            new_value = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            param.assign(new_value)
