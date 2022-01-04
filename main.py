from neural_network import NeuralNetwork
from main_functions import get_datasets, plot

LAYER_PARAMS = [64, 48, 16, 10]
LEARNING_RATE = 0.22

EPOCHS = 40

VALIDATION_PART = 0.2
TEST_PART = 0.2

if __name__ == "__main__":
    net = NeuralNetwork(LAYER_PARAMS, LEARNING_RATE)
    datasets = get_datasets(VALIDATION_PART, TEST_PART)
    train_accs = net.train(datasets[0][0], datasets[0][1],
                           datasets[1][0], datasets[1][1],
                           EPOCHS)
    test_acc = net.calc_acc(datasets[2][0], datasets[2][1])
    print(f'Test accuracy: {test_acc} ')

    plot(train_accs, test_acc, EPOCHS)
