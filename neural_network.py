import numpy as np
from typing import List
from numpy.typing import ArrayLike


class Layer:
    def __init__(self, size1: int, size2: int):
        self.size = size1
        self.weights = np.random.randn(size2, size1) * np.sqrt(1. / size2)
        self.new_weights = None
        self.activations = None


class NeuralNetwork:
    def __init__(self, sizes: List, epochs: int, learning_rate: float):
        self.epochs = epochs
        self.lrn_rate = learning_rate
        self.sizes = sizes
        self.layers = []
        self.layers.append(Layer(self.sizes[0], self.sizes[1]))
        for i in range(len(self.sizes) - 1):
            self.layers.append(Layer(self.sizes[i], self.sizes[i + 1]))

    def activation(self, x: float) -> float:
        return 1/(1 + np.exp(-x))

    def derivative(self, x: float) -> float:
        return (np.exp(-x))/((np.exp(-x) + 1) ** 2)

    def forward_pass(self, x_train: ArrayLike):
        self.layers[0].activations = x_train
        for i in range(1, len(self.layers)):
            self.layers[i].new_weights = np.dot(self.layers[i].weights,
                                                self.layers[i - 1].activations)
            self.layers[i].activations = self.activation(
                self.layers[i].new_weights)
        return self.layers[len(self.layers) - 1].activations  # jaki to typ?

    def backward_pass(self, y_train: ArrayLike, output: ArrayLike):
        change = {}
        error = 2 * (output - y_train) / output.shape[0] * self.derivative(
            self.layers[len(self.layers) - 1].new_weights)

        change[len(self.layers) - 1] = np.outer(error, self.layers[
            len(self.layers) - 2].activations)

        for i in range(len(self.layers) - 2, 0, -1):
            error = np.dot(self.layers[i + 1].weights.T, error) * self.derivative(
                self.layers[i].new_weights)

            change[i] = np.outer(error, self.layers[i - 1].activations)
        return change  # jaki to typ?

    def descent_update(self, changes: dict) -> None:
        # gradient
        for key, val in changes.items():
            self.layers[key].weights -= self.lrn_rate * val

    def calc_acc(self, x_val: ArrayLike, y_val: ArrayLike) -> float:
        predictions = []
        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))
        return np.mean(predictions)

    def train(self, x_train: ArrayLike, y_train: ArrayLike,
              x_val: ArrayLike, y_val: ArrayLike) -> None:
        for i in range(self.epochs):
            for x, y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes = self.backward_pass(y, output)
                self.descent_update(changes)
            accuracy = self.calc_acc(x_val, y_val)
            print(f'Epoch {i}: accuracy: {accuracy}')