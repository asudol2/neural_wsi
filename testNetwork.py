import sklearn.datasets as skl
from keras.utils.np_utils import to_categorical

# showing actual images (in the example just the first picture, for others change the index inside images[index])
# import matplotlib.pyplot as plt
# dataset = skl.load_digits()
# plt.gray()
# plt.matshow(dataset.images[0])
# plt.show()


if __name__ == "__main__":
    dataset = skl.load_digits().data
    # change image data from [0, 255] to [0, 1]
    dataset = dataset/255
    # represent the expected results as arrays with a 1 in the position of the result number
    # e.g. for result = 3, target array = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    target = to_categorical(skl.load_digits().target)
