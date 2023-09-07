import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


class Adaline:

    def __init__(self,
                 number_of_epochs=1000,
                 learning_rate=0.001,
                 max_error=0.1):
        self.weights = []
        self.bias = 0
        self.number_of_epochs = number_of_epochs
        self.learning_rate = learning_rate
        self.max_error = max_error

    def train(self, x, y):
        for index in range(len(x[0])):
            self.weights.append(0)

        for epoch in range(self.number_of_epochs):
            x_data = pd.DataFrame(x)

            y_train = np.dot(x_data, self.weights) + self.bias
            errors = y - y_train
            self.weights += self.learning_rate * x_data.T.dot(errors)
            self.bias += self.learning_rate * errors.sum()

            if mean_squared_error(y, y_train) <= self.max_error:
                return

    def test(self, x):
        sum = np.dot(x, self.weights) + self.bias
        return self.degree(sum)

    def degree(self, result):
        return 1 if result >= 0 else -1
