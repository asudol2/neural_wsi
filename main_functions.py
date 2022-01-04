from numpy.typing import ArrayLike
from typing import List, Tuple
import sklearn.datasets as skl
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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


def plot(train_accs: List, test_acc: float, epochs: int) -> None:

    plt.xlabel("Epoki")
    plt.ylabel("Dokładność")
    plt.title("Trenowanie sieci neuronowej")

    plt.plot([_ for _ in range(1, epochs + 1)], train_accs,
             'go--', color='g', linewidth=2)
    plt.hlines(test_acc, 2, epochs - 2, colors='r')

    plt.show()
