from numpy.typing import ArrayLike
import sklearn.datasets as skl
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from neural_network import NeuralNetwork

# showing actual images (in the example just the first picture,
# for others change the index inside images[index])
# import matplotlib.pyplot as plt
# dataset = skl.load_digits()
# plt.gray()
# plt.matshow(dataset.images[0])
# plt.show()


def get_normal_data():  # czemu typ returna wysypuje kolorowanie składni?
    """Get data to learn for net

    Returns:
        Bunch : normalized dataset
    """
    return skl.load_digits().data/16


def get_actual_nums() -> ArrayLike:
    """Get array of 'neurons' with expected values

    Returns:
        ArrayLike
    """
    return to_categorical(skl.load_digits().target)


if __name__ == "__main__":
    data = get_normal_data()
    actual_nums = get_actual_nums()

    net = NeuralNetwork([64, 32, 16, 10], 70, 0.1)
    # ^ostatnie to krok w gradiencie
    x_train, x_val, y_train, y_val = train_test_split(
        data, actual_nums, test_size=0.2)

    net.train(x_train, y_train, x_val, y_val)
