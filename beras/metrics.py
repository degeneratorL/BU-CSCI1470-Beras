import numpy as np

from beras.core import Callable


class CategoricalAccuracy(Callable):
    def forward(self, probs, labels):
        ## TODO: Compute and return the categorical accuracy of your model 
        ## given the output probabilities and true labels. 
        ## HINT: Argmax + boolean mask via '=='
        #return NotImplementedError
        pred = np.argmax(probs, axis=-1)   # shape: (batch_size,)
        true = np.argmax(labels, axis=-1)  # shape: (batch_size,)
        return np.mean(pred == true)
