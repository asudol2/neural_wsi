import numpy as np


class Layer:
    def __init__(self, size1, size2):
        self.size = size1
        self.weights = np.random.randn(size2, size1) * np.sqrt(1. / size2)
        self.newWeights = None
        self.activations = None


class NeuralNetwork:
    def __init__(self, sizes, epochs, learningRate):
        self.epochs = epochs
        self.lRate = learningRate
        self.sizes = sizes
        self.layers = []
        self.layers.append(Layer(self.sizes[0], self.sizes[1]))
        for i in range(len(self.sizes) - 1):
            self.layers.append(Layer(self.sizes[i], self.sizes[i + 1]))

    def activation(self, x):
        return 1/(1 + np.exp(-x))

    def derivative(self, x):
        return (np.exp(-x))/((np.exp(-x) + 1) ** 2)

    def forwardPass(self, xTrain):
        self.layers[0].activations = xTrain
        for i in range(1, len(self.layers)):
            self.layers[i].newWeights = np.dot(self.layers[i].weights, self.layers[i - 1].activations)
            self.layers[i].activations = self.activation(self.layers[i].newWeights)
        return self.layers[len(self.layers) - 1].activations

    def backwardPass(self, yTrain, output):
        change = {}
        error = 2 * (output - yTrain) / output.shape[0] * self.derivative(self.layers[len(self.layers) - 1].newWeights)
        change[len(self.layers) - 1] = np.outer(error, self.layers[len(self.layers) - 2].activations)
        for i in range(len(self.layers) - 2, 0, -1):
            error = np.dot(self.layers[i + 1].weights.T, error) * self.derivative(self.layers[i].newWeights)
            change[i] = np.outer(error, self.layers[i - 1].activations)
        return change

    def sgdUpdate(self, changes):
        for key, val in changes.items():
            self.layers[key].weights -= self.lRate * val

    def calcAccuracy(self, xVal, yVal):
        predictions = []
        for x, y in zip(xVal, yVal):
            output = self.forwardPass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))
        return np.mean(predictions)

    def train(self, xTrain, yTrain, xVal, yVal):
        for i in range(self.epochs):
            for x, y in zip(xTrain, yTrain):
                output = self.forwardPass(x)
                changes = self.backwardPass(y, output)
                self.sgdUpdate(changes)
            accuracy = self.calcAccuracy(xVal, yVal)
            print(f'Epoch {i}: accuracy: {accuracy}')
