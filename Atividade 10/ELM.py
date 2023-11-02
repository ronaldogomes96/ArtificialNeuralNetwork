import numpy as np


class ELM:

    def __init__(self, number_of_neurons=10):
        self.number_of_neurons = number_of_neurons
        self.weights_input_hidden = None
        self.bias_input_hidden = None
        self.weights_hidden_output = None

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X, y):
        number_of_features = X.shape[1]

        self.weights_input_hidden = np.random.randn(number_of_features, self.number_of_neurons)
        self.bias_input_hidden = np.random.randn(self.number_of_neurons)

        input_sum = np.dot(X, self.weights_input_hidden) + self.bias_input_hidden

        hidden_layer_activations = self.sigmoid(input_sum)
        hidden_layer_activations_inverse = np.linalg.pinv(hidden_layer_activations)

        self.weights_hidden_output = np.dot(hidden_layer_activations_inverse, y)

    def predict(self, X):
        input_sum = np.dot(X, self.weights_input_hidden) + self.bias_input_hidden
        hidden_layer_activations = self.sigmoid(input_sum)

        predictions = np.dot(hidden_layer_activations, self.weights_hidden_output)

        return np.round(predictions)

