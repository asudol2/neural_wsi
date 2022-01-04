from numpy.typing import ArrayLike
from typing import List, Tuple

import sklearn.datasets as skl
import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split


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
    """Get train, validation and test datasets

    Args:
        val_part (float): part of all data to be validational set
        test_part (float): part of all data to be test set

    Returns:
        List[Tuple[ArrayLike]]: train, val and test datasets
            in tuples like (input, output)
    """

    data = get_normal_data()
    actual_nums = get_actual_nums()
    x_train, x_val, y_train, y_val = train_test_split(
        data, actual_nums, test_size=val_part + test_part
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_val, y_val, test_size=test_part/(1-val_part-test_part)
    )
    return [(x_train, y_train), (x_val, y_val), (x_test, y_test)]


def plot(train_accs: List, test_acc: float, epochs: int) -> None:
    """Generate plot of accuracies for epochs with line of test accuracy

    Args:
        train_accs (List): accuracies in train stage
        test_acc (float): accuracy in test stage
        epochs (int): number of epochs
    """

    plt.xlabel("Epoki")
    plt.ylabel("Dokładność")
    plt.title("Trenowanie sieci neuronowej")

    plt.plot([_ for _ in range(1, epochs + 1)], train_accs,
             'go--', color='g', linewidth=2)
    plt.hlines(test_acc, 2, epochs - 2, colors='r')

    plt.savefig('network_training.png')
    plt.show()
