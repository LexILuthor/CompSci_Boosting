from settings import Settings
from gradient_boost_model import GradientBoostModel
from fashion_mnist_master.utils import mnist_reader
import data.read_data as rd
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
from os import path


def plot_results(x_value, y1_value, y2_value=None, x_label="", y_axis_label="", y1_legend="", y2_legend="",
                 integer_ticks=False, plot_name=""):
    plt.style.use('ggplot')
    # ['fivethirtyeight', 'seaborn-pastel', 'seaborn-whitegrid', 'ggplot', 'grayscale']

    fig, ax = plt.subplots()

    ax.plot(x_value, y1_value, linewidth=2.0, label=y1_legend, color="blue")
    if y2_value is not None:
        ax.plot(x_value, y2_value, linewidth=2.0, label=y2_legend, color="orange")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_axis_label)
    if integer_ticks:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.set(xlim=(0, 8), xticks=np.arange(1, 8), ylim=(0, 8), yticks=np.arange(1, 8))

    i = 0
    while path.exists("plots/" + plot_name + str(i) + ".png"):
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
    plot_results(n_estimators_vector, accuracy, x_label="number of estimators", y_axis_label="accuracy",
                 plot_name="n_estimators")
    print("number of estimators\n", n_estimators_vector)
    print("accuracy\n", accuracy)


def testing_learning_rate(learning_rate_vector, x_train, y_train, x_test, y_test, max_depth_value=2):
    accuracy = []
    for learning_rate in learning_rate_vector:
        settings = Settings(learning_rate=learning_rate, n_estimators=20, max_depth=max_depth_value)

        gradient_boost_model = GradientBoostModel(settings)

        gradient_boost_model.train(x_train, y_train)
        # results = gradient_boost_model.evaluate(x_test, y_test)
        error_rate = gradient_boost_model.error_rate(x_test, y_test)
        accuracy.append(1 - error_rate)

    accuracy = np.array(accuracy)
    plot_results(learning_rate_vector, accuracy, x_label="learning rate", y_axis_label="accuracy",
                 plot_name="learning_rate")
    print("learning rate\n", learning_rate_vector)
    print("accuracy\n", accuracy)


def testing_max_depth(max_depth_vector, x_train, y_train, x_test, y_test, n_estimators=20):
    accuracy = []

    for max_depth in max_depth_vector:
        settings = Settings(max_depth=max_depth, n_estimators=n_estimators)

        gradient_boost_model = GradientBoostModel(settings)

        gradient_boost_model.train(x_train, y_train)
        # results = gradient_boost_model.evaluate(x_test, y_test)
        error_rate = gradient_boost_model.error_rate(x_test, y_test)
        accuracy.append(1 - error_rate)

    accuracy = np.array(accuracy)
    plot_results(max_depth_vector, accuracy, x_label="maximum tree depth", y_axis_label="accuracy", integer_ticks=True,
                 plot_name="max_depth")
    print("maximum tree depth\n", max_depth_vector)
    print("accuracy\n", accuracy)


def test_gradient_boosting(x_train, y_train, x_test, y_test, settings):
    gradient_boost_model = GradientBoostModel(settings)

    gradient_boost_model.train(x_train, y_train)

    # results = gradient_boost_model.evaluate(x_test, y_test)
    error_rate_train = gradient_boost_model.error_rate(x_train, y_train)
    error_rate_test = gradient_boost_model.error_rate(x_test, y_test)

    accuracy_train = 1 - error_rate_train
    accuracy_test = 1 - error_rate_test

    return accuracy_train, accuracy_test


def test_and_plot(x_train, y_train, x_test, y_test,
                  n_estimators=None,
                  learning_rate=None,
                  max_depth=None,
                  random_state=None):
    if n_estimators is None:
        n_estimators = []
    if learning_rate is None:
        learning_rate = []
    if random_state is None:
        random_state = []
    if max_depth is None:
        max_depth = []

    accuracy_train = []
    accuracy_test = []
    for n in n_estimators:
        settings = Settings(n_estimators=n)
        accuracy_train_value, accuracy_test_value = test_gradient_boosting(x_train, y_train, x_test, y_test, settings)
        accuracy_train.append(accuracy_train_value)
        accuracy_test.append(accuracy_test_value)

    plot_results(n_estimators, accuracy_train, accuracy_test, x_label="number of estimators", plot_name="n_estimators",
                 y1_legend="train accuracy", y2_legend="test accuracy", y_axis_label="accuracy")
    print("number of estimators\n", n_estimators)
    print("train accuracy\n", accuracy_train)
    print("test accuracy\n", accuracy_test)

    accuracy_train = []
    accuracy_test = []
    for depth in max_depth:
        settings = Settings(max_depth=depth)
        accuracy_train_value, accuracy_test_value = test_gradient_boosting(x_train, y_train, x_test, y_test, settings)
        accuracy_train.append(accuracy_train_value)
        accuracy_test.append(accuracy_test_value)

    plot_results(n_estimators, accuracy_train, accuracy_test, x_label="max depth", plot_name="max_depth",
                 y1_legend="train accuracy", y2_legend="test accuracy", y_axis_label="accuracy")
    print("max depth\n", max_depth)
    print("train accuracy\n", accuracy_train)
    print("test accuracy\n", accuracy_test)

    accuracy_train = []
    accuracy_test = []
    for rate in learning_rate:
        settings = Settings(learning_rate=rate)
        accuracy_train_value, accuracy_test_value = test_gradient_boosting(x_train, y_train, x_test, y_test, settings)
        accuracy_train.append(accuracy_train_value)
        accuracy_test.append(accuracy_test_value)

    plot_results(n_estimators, accuracy_train, accuracy_test, x_label="learning rate", plot_name="learning_rate",
                 y1_legend="train accuracy", y2_legend="test accuracy", y_axis_label="accuracy")
    print("learning rate\n", learning_rate)
    print("train accuracy\n", accuracy_train)
    print("test accuracy\n", accuracy_test)


def main():
    x_train, y_train, x_test, y_test, x_final_test, y_final_test = rd.read_data()

    n_estimators_vector = np.array([1, 10, 20, 40, 60, 80, 100])

    learning_rate_vector = [0.1, 0.3, 0.5, 0.7, 0.9]

    max_depth_vector = np.array([1, 2, 3, 4])

    test_and_plot(x_train, y_train, x_test, y_test,
                  n_estimators=n_estimators_vector,
                  learning_rate=learning_rate_vector,
                  max_depth=max_depth_vector,
                  random_state=None)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
