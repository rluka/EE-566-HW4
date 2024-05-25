import numpy as np

import torch.nn as nn
from torchvision import datasets

# Activation functions and their derivatives

def sigmoid(z: float) -> float:
    return np.exp(z) / (1 + np.exp(z))

def d_sigmoid(z: float) -> float:
    return sigmoid(z) * (1 - sigmoid(z))

def tanh(z: float) -> float:
    return np.tanh(z)

def d_tanh(z: float) -> float:
    return 1 - np.tanh(z) ** 2

def ReLU(z: float) -> float:
    return np.maximum(0, z)

def d_ReLU(z: float) -> float:
    return (ReLU(z) > 0) * 1.0

def softmax(z: np.ndarray) -> np.ndarray:
    return np.exp(z) / np.sum(np.exp(z))

def cross_entropy_loss(gamma_hat: np.ndarray, gamma: np.ndarray, rho: float, weights) -> float:
    regularization = rho * sum([np.linalg.norm(w, ord='fro') ** 2 for w in weights])
    unregularized_loss = -np.dot(gamma, np.log(gamma_hat))
    return unregularized_loss + regularization


# Preprocessing the data

def load_and_preprocess_data(path: str):
    # Load the training and testing dataset
    train_data = datasets.CIFAR10(root=path, train=True, download=True)
    test_data = datasets.CIFAR10(root=path, train=False, download=True)

    # Flatten the arrays
    train_set = train_data.data.reshape(train_data.data.shape[0], -1)
    train_labels = train_data.targets

    test_set = test_data.data.reshape(test_data.data.shape[0], -1)
    test_labels = test_data.targets

    # Scale them down to [0, 1]
    train_set = train_set / 255.0
    test_set = test_set / 255.0

    # Use z-score normalization
    train_set = (train_set - np.mean(train_set, axis=0)) / np.std(train_set, axis=0)
    test_set = (test_set - np.mean(test_set, axis=0)) / np.std(test_set, axis=0)

    return train_set, train_labels, test_set, test_labels

def transform_label(gamma: int) -> np.ndarray:
    gamma_vector = np.zeros(10)
    gamma_vector[gamma] = 1
    return gamma_vector

# A neural network wtih 4 layers -> 1 input layer, 2 hidden layers, 1 output layer
class NeuralNetwork:
    def __init__(
            self, 
            input_size: int, 
            hidden_size: int, 
            output_size: int, 
            activation_fcn, 
            d_activation_fcn,
            learning_rate: float,
            rho: float,
            p_dropout: np.ndarray = None
        ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation_fcn
        self.d_activation = d_activation_fcn
        self.mu = learning_rate
        self.rho = rho
        self.p_dropout = p_dropout

        # Initialize weights and biases
        self.W1 = np.random.uniform(low=-1/np.sqrt(self.input_size), high=1/np.sqrt(self.input_size), size=(self.input_size, self.hidden_size))
        self.W2 = np.random.uniform(low=-1/np.sqrt(self.hidden_size), high=1/np.sqrt(self.hidden_size), size=(self.hidden_size, self.hidden_size))
        self.W3 = np.random.uniform(low=-1/np.sqrt(self.hidden_size), high=1/np.sqrt(self.hidden_size), size=(self.hidden_size, self.output_size))
        self.b1 = np.random.randn(self.hidden_size)
        self.b2 = np.random.randn(self.hidden_size)
        self.b3 = np.random.randn(self.output_size)

    def xavier_init(self, n_in, n_out):
        sigma = np.sqrt(2 / (n_in + n_out))
        return np.random.normal(0, sigma, (n_in, n_out))

    # Return model weights
    def weights(self):
        return [self.W1, self.W2, self.W3]
    
    # One forward pass of a sample
    def forward(self, input):

        self.input = input

        if not self.p_dropout:
            self.a1 = np.ones(self.input_size)
            self.a2 = np.ones(self.hidden_size)
            self.a3 = np.ones(self.hidden_size)
        else:
            self.a1 = np.random.choice([0, 1], size=self.input_size, p=[self.p_dropout[0], 1 - self.p_dropout[0]])
            self.a2 = np.random.choice([0, 1], size=self.hidden_size, p=[self.p_dropout[1], 1 - self.p_dropout[1]])
            self.a3 = np.random.choice([0, 1], size=self.hidden_size, p=[self.p_dropout[2], 1 - self.p_dropout[2]])

        # Input - 1st hidden
        self.z2 = self.W1.T @ np.multiply(self.input, self.a1) - self.b1
        self.y2 = self.activation(self.z2)
        # 1st hidden - 2nd hidden
        self.z3 = self.W2.T @ np.multiply(self.y2, self.a2) - self.b2
        self.y3 = self.activation(self.z3)
        # 2nd hidden - output
        self.z = self.W3.T @ np.multiply(self.y3, self.a3) - self.b3
        gamma_hat = softmax(self.z)
        
        return gamma_hat
    
    # One backward pass of a sample
    def backward(self, gamma_hat, gamma):
        # Regularized cross-entropy loss + softmax
        delta_output = gamma_hat - gamma
        
        # Output - 2nd hidden
        W3_old = self.W3.copy()
        self.W3 = (1 - 2*self.mu*self.rho) * W3_old - self.mu * np.outer(np.multiply(self.y3, self.a3), delta_output)
        self.b3 = self.b3 + self.mu * delta_output
        delta_3 = np.multiply(np.multiply(W3_old @ delta_output, self.d_activation(self.z3)), self.a3)

        # 2nd hidden - 1st hidden
        W2_old = self.W2.copy()
        self.W2 = (1 - 2*self.mu*self.rho) * W2_old - self.mu * np.outer(np.multiply(self.y2, self.a2), delta_3)
        self.b2 = self.b2 + self.mu * delta_3
        delta_2 = np.multiply(np.multiply(W2_old @ delta_3, self.d_activation(self.z2)), self.a2)

        # 1st hidden - input
        W1_old = self.W1.copy()
        self.W1 = (1 - 2*self.mu*self.rho) * W1_old - self.mu * np.outer(np.multiply(self.input, self.a1), delta_2)
        self.b1 = self.b1 + self.mu * delta_2

    def normalize_weights(self):
        if self.p_dropout:
            self.W1 = (1 - self.p_dropout[0]) * self.W1
            self.W2 = (1 - self.p_dropout[1]) * self.W2
            self.W3 = (1 - self.p_dropout[2]) * self.W3
            self.b1 = (1 - self.p_dropout[0]) * self.b1
            self.b2 = (1 - self.p_dropout[1]) * self.b2
            self.b3 = (1 - self.p_dropout[2]) * self.b3