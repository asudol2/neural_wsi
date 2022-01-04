import numpy as np
from typing import List
from numpy.typing import ArrayLike


class Layer:
    def __init__(self, size: int, norm_value: int):
        self.size = size
        self.weights = np.random.randn(
            norm_value, size) * np.sqrt(1. / norm_value)
        self.new_weights = None
        self.activations = None


class NeuralNetwork:

    def __init__(self, sizes: List, learning_rate: float):
        """Construct NeuralNetwork class instant

        Args:
            sizes (List): list of numbers of neurons in succeesing layers
            learning_rate (float): in other words, step in gradient descent
        """

        self.lrn_rate = learning_rate
        self.sizes = sizes
        self.layers = []
        self.layers.append(Layer(self.sizes[0], self.sizes[1]))
        for i in range(len(self.sizes) - 1):
            self.layers.append(Layer(self.sizes[i], self.sizes[i + 1]))

    def activation(self, x: float) -> float:
        """Function representing sigmoid function

        Args:
            x (float)

        Returns:
            float : normalized arg
        """

        return 1/(1 + np.exp(-x))

    def derivative(self, x: float) -> float:
        """Derivative of sigmoid function

        Args:
            x (float)

        Returns:
            float
        """

        return (np.exp(-x))/((np.exp(-x) + 1) ** 2)

    def forward_pass(self, x_train: ArrayLike) -> ArrayLike:
        """Get output from last layer neurons

        Args:
            x_train (ArrayLike): input args array

        Returns:
            ArrayLike: results of identificaton
        """
        self.layers[0].activations = x_train
        for i in range(1, len(self.layers)):
            self.layers[i].new_weights = np.dot(self.layers[i].weights,
                                                self.layers[i - 1].activations)
            self.layers[i].activations = self.activation(
                self.layers[i].new_weights)
        return self.layers[len(self.layers) - 1].activations

    def backward_pass(self, y_train: ArrayLike, output: ArrayLike) -> dict:
        """Teach network.

        Args:
            y_train (ArrayLike): expected output
            output (ArrayLike): actual output (from forward pass)

        Returns:
            dict: array of changes for gradient descent
        """
        change = {}
        error = 2 * (output - y_train) / output.shape[0] * self.derivative(
            self.layers[len(self.layers) - 1].new_weights)

        change[len(self.layers) - 1] = np.outer(error, self.layers[
            len(self.layers) - 2].activations)

        for i in range(len(self.layers) - 2, 0, -1):
            error = np.dot(self.layers[i + 1].weights.T, error) * self.derivative(
                self.layers[i].new_weights)

            change[i] = np.outer(error, self.layers[i - 1].activations)
        return change

    def descent_update(self, changes: dict) -> None:
        """Update gradient descent array

        Args:
            changes (dict)
        """

        for key, val in changes.items():
            self.layers[key].weights -= self.lrn_rate * val

    def calc_acc(self, x_val: ArrayLike, y_val: ArrayLike) -> float:
        """Calculate accuracy of trained net

        Args:
            x_val (ArrayLike): input data
            y_val (ArrayLike): expected output data

        Returns:
            float
        """

        predictions = []
        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))
        return round(np.mean(predictions), 3)

    def train(self, x_train: ArrayLike, y_train: ArrayLike,
              x_val: ArrayLike, y_val: ArrayLike, epochs: int) -> List[float]:
        """Conduct all training of network.
        Print current accuracy.

        Args:
            x_train (ArrayLike): input of learning dataset
            y_train (ArrayLike): output of learning dataset
            x_val (ArrayLike): input of validation dataset
            y_val (ArrayLike): output of validation dataset
            epochs (int): number of epochs we want to train net

        Returns:
            List[float]: list of accuracies through epochs
        """
        accs = []
        for i in range(epochs):
            for x, y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes = self.backward_pass(y, output)
                self.descent_update(changes)
            acc = self.calc_acc(x_val, y_val)
            accs.append(acc)
            print(f'Epoch {i}: accuracy: {acc}')
        return accs
