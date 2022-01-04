from numpy.typing import ArrayLike
from typing import List, Tuple
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


def get_normal_data() -> dict:
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


def get_datasets(val_part: float, test_part: float) -> List[Tuple[ArrayLike]]:
    data = get_normal_data()
    actual_nums = get_actual_nums()
    x_train, x_val, y_train, y_val = train_test_split(
        data, actual_nums, test_size=val_part + test_part
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_val, y_val, test_size=test_part/(1-val_part-test_part)
    )
    return [(x_train, y_train), (x_val, y_val), (x_test, y_test)]


LayerParams = [64, 32, 16, 10]
Epochs = 70
LearningRate = 0.1  # krok w gradiencie

if __name__ == "__main__":
    net = NeuralNetwork(LayerParams, Epochs, LearningRate)
    datasets = get_datasets(0.3, 0.2)
    net.train(datasets[0][0], datasets[0][1],
              datasets[1][0], datasets[1][1])
