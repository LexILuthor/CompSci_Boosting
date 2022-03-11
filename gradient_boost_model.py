from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier


class GradientBoostModel:
    def __init__(self, se):
        self.gradient_booster = GradientBoostingClassifier(n_estimators=se.n_estimators, learning_rate=se.learning_rate,
                                                           max_depth=se.max_depth,
                                                           random_state=se.random_state)

    def train(self, x_train, y_train):
        self.gradient_booster.fit(x_train, y_train)

    def evaluate(self, x_test, y_test):
        return classification_report(y_test, self.gradient_booster.predict(x_test))
