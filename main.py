from settings import Settings
from gradient_boost_model import GradientBoostModel
from fashion_mnist_master.utils import mnist_reader


def main():
    settings = Settings(n_estimators=50)

    gradient_boost_model = GradientBoostModel(settings)

    x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    gradient_boost_model.train(x_train, y_train)
    results = gradient_boost_model.evaluate(x_test, y_test)
    print(results)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
