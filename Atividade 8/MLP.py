import numpy as np
import pandas as pd
from Statistics import tanh, tanh_derivative, sigmoid, sigmoid_derivative


class MLP:

    def __init__(self,
                 number_of_epochs=10000,
                 hidden_layer_neurons=4,
                 learning_rate=0.1,
                 max_error=0.1,
                 activation_function='hyperbolic_tangent'):
        self.weights = []
        self.bias = 0
        self.number_of_epochs = number_of_epochs
        self.learning_rate = learning_rate
        self.max_error = max_error
        self.hidden_layer_neurons = hidden_layer_neurons
        self.activation_function = activation_function

    def predict(self, X_test):
        if self.activation_function == 'hyperbolic_tangent':
            _, result = self.feed_forward_propagation(X_test)
            return 0 if result[0] == 0 else 1

        elif self.activation_function == 'sigmoid':
            result = sigmoid(np.dot(sigmoid(np.dot(X_test, self.weights[0])), self.weights[1]))
            return 1 if result[0] >= 0.5 else 0

    def fit(self, X, y):
        self.weights.append(np.random.uniform(-1, 1, (X.shape[1], self.hidden_layer_neurons)))
        self.weights.append(np.random.uniform(-1, 1, (self.hidden_layer_neurons, y.shape[1])))

        for _ in range(self.number_of_epochs):
            hidden_layer_output, output_layer_output = self.feed_forward_propagation(X)

            self.back_propagation(X, y, hidden_layer_output, output_layer_output)

    def feed_forward_propagation(self, X):
        if self.activation_function == 'hyperbolic_tangent':
            hidden_layer_output = tanh(np.dot(X, self.weights[0]))
            output_layer_output = tanh(np.dot(hidden_layer_output, self.weights[1]))

        elif self.activation_function == 'sigmoid':
            hidden_layer_output = sigmoid(np.dot(X, self.weights[0]))
            output_layer_output = sigmoid(np.dot(hidden_layer_output, self.weights[1]))

        return hidden_layer_output, output_layer_output

    def back_propagation(self, X, y, hidden_layer_output, output_layer_output):
        output_error = y - output_layer_output

        if self.activation_function == 'hyperbolic_tangent':
            delta_output = output_error * tanh_derivative(output_layer_output)

        elif self.activation_function == 'sigmoid':
            delta_output = output_error * sigmoid_derivative(output_layer_output)

        hidden_layer_error = delta_output.dot(self.weights[1].T)

        if self.activation_function == 'hyperbolic_tangent':
            delta_hidden_layer = hidden_layer_error * tanh_derivative(hidden_layer_output)

        elif self.activation_function == 'sigmoid':
            delta_hidden_layer = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

        self.weights[0] += X.T.dot(delta_hidden_layer) * self.learning_rate
        self.weights[1] += hidden_layer_output.T.dot(delta_output) * self.learning_rate
