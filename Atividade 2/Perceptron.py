import numpy as np
import pandas as pd


class Perceptron:

    def __init__(self,
                 number_of_epochs=1000,
                 learning_rate=0.1):
        self.weights = []
        self.bias = np.array([-1])
        self.number_of_epochs = number_of_epochs
        self.learning_rate = learning_rate

    def train(self, x, y):
        for index in range(len(x[0]) + 1):
            self.weights.append(0.5)

        for epoch in range(self.number_of_epochs):
            number_of_erros = 0
            new_X = pd.DataFrame(x)

            for index, x_row in new_X.iterrows():
                x_row = np.array(x_row)
                x_row = np.append(x_row, self.bias[0])

                sum = np.dot(x_row, self.weights)

                y_train = 1 if sum >= 0 else 0

                network_error = y[index] - y_train

                if y[index] != y_train:
                    for weight_index, weight in enumerate(self.weights):
                        self.weights[weight_index] = weight + (self.learning_rate * network_error * x_row[weight_index])
                    number_of_erros += 1

            if number_of_erros == 0:
                return

    def test(self, x):
        x.append(self.bias[0])
        sum = np.dot(x, self.weights)
        return 1 if sum >= 0 else 0
