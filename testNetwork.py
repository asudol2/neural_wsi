import sklearn.datasets as skl
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from neuralNetwork import NeuralNetwork

# showing actual images (in the example just the first picture, for others change the index inside images[index])
# import matplotlib.pyplot as plt
# dataset = skl.load_digits()
# plt.gray()
# plt.matshow(dataset.images[0])
# plt.show()


if __name__ == "__main__":
    dataset = skl.load_digits().data
    # change image data from [0, 16] to [0, 1]
    dataset = dataset/16
    # represent the expected results as arrays with a 1 in the position of the result number
    # e.g. for result = 3, target array = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    target = to_categorical(skl.load_digits().target)
    nn = NeuralNetwork([64, 32, 16, 10], 70, 0.1)
    xTrain, xVal, yTrain, yVal = train_test_split(dataset, target, test_size=0.2)
    nn.train(xTrain, yTrain, xVal, yVal)
