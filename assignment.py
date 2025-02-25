from types import SimpleNamespace
from beras.activations import ReLU, LeakyReLU, Softmax
from beras.layers import Dense
from beras.losses import CategoricalCrossEntropy, MeanSquaredError
from beras.metrics import CategoricalAccuracy
from beras.onehot import OneHotEncoder
from beras.optimizers import Adam
from preprocess import load_and_preprocess_data
import numpy as np

from beras.model import SequentialModel

def get_model():
    """
    Task 1:
    Create a SequentialModel with the layers you want.
    For MNIST (28*28 = 784 features) -> 10 classes is typical.
    We'll do:
      Dense(784, 128), ReLU,
      Dense(128, 10),  Softmax
    """
    model = SequentialModel(
        [
            Dense(784, 128, initializer="xavier"),  # or "kaiming"/"normal"
            ReLU(),
            Dense(128, 10, initializer="xavier"),
            Softmax()
        ]
    )
    return model

def get_optimizer():
    """
    Task 2:
    Choose an optimizer. We'll use Adam with a certain learning rate.
    """
    return Adam(learning_rate=0.01)

def get_loss_fn():
    """
    Task 3:
    Choose a loss function.
    We'll use CategoricalCrossEntropy for classification.
    """
    return CategoricalCrossEntropy()

def get_acc_fn():
    """
    Task 4:
    Choose an accuracy metric. We'll use CategoricalAccuracy.
    """
    return CategoricalAccuracy()

if __name__ == '__main__':

    # 1. Create a SequentialModel using get_model
    model = get_model()

    # 2. Compile the model with optimizer, loss function, and accuracy metric
    optimizer = get_optimizer()
    loss_fn = get_loss_fn()
    acc_fn = get_acc_fn()
    model.compile(optimizer, loss_fn, acc_fn)

    # 3. Load and preprocess the data (flatten + normalize -> Tensors)
    train_inputs, train_labels, test_inputs, test_labels = load_and_preprocess_data()

    # Convert labels to one-hot
    # For classification, we want labels as one-hot encoded
    encoder = OneHotEncoder()
    train_labels_onehot = encoder(train_labels)  # shape: (num_train_samples, 10)
    test_labels_onehot  = encoder(test_labels)   # shape: (num_test_samples, 10)

    # 4. Train the mode
    EPOCHS = 10
    BATCH_SIZE = 256
    history = model.fit(train_inputs, train_labels_onehot, epochs=EPOCHS, batch_size=BATCH_SIZE)
    print("\nTraining history:", history)

    # 5. Evaluate the model on test set
    eval_stats = model.evaluate(test_inputs, test_labels_onehot, batch_size=BATCH_SIZE)
    print("\nFinal Evaluation on Test Data:", eval_stats)