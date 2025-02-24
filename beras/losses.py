import numpy as np

from beras.core import Diffable, Tensor

import tensorflow as tf


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return np.mean((y_true - y_pred) ** 2) #NotImplementedError

    def get_input_gradients(self) -> list[Tensor]:
        y_pred, y_true = self.inputs  # The two inputs to forward(...)
        # number of total elements in y_pred => batch_size * num_features
        b_n = y_pred.size

        # partial wrt y_pred
        grad_pred = (2.0 * (y_pred - y_true)) / b_n
        
        # partial wrt y_true => treat as constant => 0
        grad_true = np.zeros_like(y_true)

        return [grad_pred, grad_true]

class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        eps = 1e-9
        p = np.clip(y_pred, eps, 1 - eps)
        # sum across classes, shape => (batch_size,)
        sample_losses = -np.sum(y_true * np.log(p), axis=1)
        # average across batch => scalar
        return np.mean(sample_losses)

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        y_pred, y_true = self.inputs  # The two inputs
        eps = 1e-9
        p = np.clip(y_pred, eps, 1 - eps)

        batch_size = p.shape[0]

        # derivative of -sum(y_true * log(p)) wrt p => -y_true / p
        # and we also average across batch, so divide by batch_size
        grad_pred = - (y_true / p) / batch_size

        grad_true = np.zeros_like(y_true)

        return [grad_pred, grad_true]
