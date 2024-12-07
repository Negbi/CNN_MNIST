import random
import numpy as np


# Define the neural network class
class Network(object):
    # initialize the neural network with given layer sizes
    def __init__(self, sizes, patience=5):
        self.num_layers = len(sizes)  # the total number of layers
        self.sizes = sizes
        # initializing the biases and weights randomly for each layer (except the input layer)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.zero_grad_count = 0
        self.j = 0
        self.patience = patience  # the number of mini_batches with gradients close to zero that we want to stop early

    def feedforward(self, a):
        # loop through each layer
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def sgd(self, training_data, rounds, mini_batch_size, l_rate):
        n = len(training_data)
        while self.j < rounds:
            # shuffle the training data
            random.shuffle(training_data)
            # split the training data
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            # update weights and biases for each mini-batch
            for mini_batch in mini_batches:
                self.calc_mini_batch(mini_batch, l_rate, rounds)
            print("round {0}".format(self.j))
            self.j = self.j + 1

    def calc_mini_batch(self, mini_batch, l_rate, rounds):
        # initialize gradients to zero
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        # loop through each example in the mini-batch
        for x, y in mini_batch:
            # calculate gradients using backpropagation
            delta_grad_b, delta_grad_w = self.backpropagation(x, y)
            # updating the gradients
            grad_b = [g_b + d_g_b for g_b, d_g_b in zip(grad_b, delta_grad_b)]
            grad_w = [g_w + d_g_w for g_w, d_g_w in zip(grad_w, delta_grad_w)]
        # update the actual weights and biases
        self.weights = [w - (l_rate / len(mini_batch)) * g_w for w, g_w in zip(self.weights, grad_w)]
        self.biases = [b - (l_rate / len(mini_batch)) * g_b for b, g_b in zip(self.biases, grad_b)]

        # checking if the gradients sre close to zero in all the layers
        if close_to_zero(grad_b) and close_to_zero(grad_w):
            self.zero_grad_count += 1
        else:
            self.zero_grad_count = 0

        # stop training early if we got the number of mini_batches with gradients close to zero
        if self.zero_grad_count >= self.patience:
            self.j = rounds

    def backpropagation(self, x, y):
        # initialize gradients to zero
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward: compute activations for each layer
        activation = x
        activations = [x]  # list of activations for each layer
        zs = []  # list to store z vectors (weighted inputs) for each layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # compute the output error of the last layer
        delta = cost_func_prime_z(zs[-1], y)
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].transpose())
        # backpropagate the error to previous layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)  # Include the sigmoid derivative
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return grad_b, grad_w


def cost_func_prime_z(z, y):
    num = 2 * y - 1
    sigmoid_val = sigmoid(num * z)
    return (sigmoid_val - 1) * num


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    num = sigmoid(z)
    return num * (1 - num)


def close_to_zero(grad, epsilon=1e-9):
    return all(np.all(np.abs(g) < epsilon) for g in grad)