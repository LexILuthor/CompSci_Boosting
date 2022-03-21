from settings import Settings
from gradient_boost_model import GradientBoostModel
from fashion_mnist_master.utils import mnist_reader
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
from os import path


def plot_results(x_value, y_value, x_label="", y_label="", integer_ticks=False, plot_name=""):
    plt.style.use('ggplot')
    # ['fivethirtyeight', 'seaborn-pastel', 'seaborn-whitegrid', 'ggplot', 'grayscale']

    fig, ax = plt.subplots()

    ax.plot(x_value, y_value, linewidth=2.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if integer_ticks:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.set(xlim=(0, 8), xticks=np.arange(1, 8), ylim=(0, 8), yticks=np.arange(1, 8))

    i = 0
    while path.exists("plots/" + plot_name + str(i)):
        i += 1

    plt.savefig("plots/" + plot_name + str(i))

    plt.show()


def testing_n_estimators(n_estimators_vector, x_train, y_train, x_test, y_test, max_depth_value=1):
    accuracy = []
    for n_estimators in n_estimators_vector:
        settings = Settings(n_estimators=n_estimators, max_depth=max_depth_value)

        gradient_boost_model = GradientBoostModel(settings)

        gradient_boost_model.train(x_train, y_train)
        # results = gradient_boost_model.evaluate(x_test, y_test)
        error_rate = gradient_boost_model.error_rate(x_test, y_test)
        accuracy.append(1 - error_rate)

    accuracy = np.array(accuracy)
    plot_results(n_estimators_vector, accuracy, x_label="number of estimators", y_label="accuracy",
                 plot_name="n_estimators")
    print("number of estimators\n", n_estimators_vector)
    print("accuracy\n", accuracy)


def testing_max_depth(max_depth_vector, x_train, y_train, x_test, y_test, n_estimators=3):
    accuracy = []

    for max_depth in max_depth_vector:
        settings = Settings(max_depth=max_depth, n_estimators=n_estimators)

        gradient_boost_model = GradientBoostModel(settings)

        gradient_boost_model.train(x_train, y_train)
        # results = gradient_boost_model.evaluate(x_test, y_test)
        error_rate = gradient_boost_model.error_rate(x_test, y_test)
        accuracy.append(1 - error_rate)

    accuracy = np.array(accuracy)
    plot_results(max_depth_vector, accuracy, x_label="maximum tree depth", y_label="accuracy", integer_ticks=True,
                 plot_name="max_depth")
    print("maximum tree depth\n", max_depth_vector)
    print("accuracy\n", accuracy)


def main():
    x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    n_estimators_vector = np.array([1, 10, 20, 40, 60, 80, 100])
    testing_n_estimators(n_estimators_vector, x_train, y_train, x_test, y_test)

    max_depth_vector = np.array([1, 2, 3, 4])
    testing_max_depth(max_depth_vector, x_train, y_train, x_test, y_test, n_estimators=1)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
