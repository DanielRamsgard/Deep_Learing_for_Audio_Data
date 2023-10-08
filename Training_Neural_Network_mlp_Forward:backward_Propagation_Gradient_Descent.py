import numpy as np
from random import random

# save activations and derivatives 
# implement backpropagation
# implement gradient descent
# implement train method (use both BP and GD)
# train our network with some dummy dataset
# make some predictions

class MLP:
    

    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_output = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_output] # number of neurons

        # initialize weights
        self.weights = []
        for i in range(len(layers) - 1): # if we have three layers: 2 w matrices because # of w matrcies is between subsequent layers
            w = np.random.rand(layers[i], layers[i+1]) # layers[i] like number of x (for # of rows); layers[i+1] like s (for # of columns)
            self.weights.append(w) # list of w matrices

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations


        derivatives = [] # number of derivatives equal to number of weights (matrices)
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives

    def back_propagate(self, error, verbose=False):
            
        # dE/dW_i = (y - a_[i+1])s(prime)(h_[i+1])a_i
        # s(prime)(h_[i+1]) = s(h_[i+1])(1 - s(h_[i+1]))
        # s(h_[i+1]) = a_[i+1]

        # dE/dW_[i-1] = (y - a_[i+1])s(prime)(h_[i+1])W_i s(prime)(h_i)a_[i-1]

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self.sigmoid_derivative(activations) # ndarry9([0.1, 0.2]) --> ndarray([[0.1, 0.2]])
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i] # ndarry9[0.1, 0.2]) --> ndarray([[0.1], [0.2]])
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)
            if verbose:
                print("Derviatves for W{}: {}".format(i, self.derivatives[i]))


        return error
    

    def gradient_descent(self, learning_rate):
            for i in range(len(self.weights)):
                weights = self.weights[i]
                derivatives = self.derivatives[i]
                weights += derivatives * learning_rate


    def train(self, inputs, targets, epochs, learning_rate):
         
        for i in range(epochs):
            sum_error = 0
            for input, target in zip(inputs, targets): 
                # perform forward prop
                output = self.forward_propagate(input)

                # calculate error
                error = target - output

                # perform back progation
                self.back_propagate(error)

                # apply gradient descent

                self.gradient_descent(learning_rate=0.1)

                sum_error += self.mse(target, output)
            # report error
            print(f"Error: {(sum_error / len(inputs))} at epoch {i}")


    def mse(self, target, output):
        return np.average((target - output) ** 2)


    def sigmoid_derivative(self, x):
        return x * (1.0 - x)


    def forward_propagate(self, inputs):
        activations = inputs
        self.activations[0] = inputs
        for i, w in enumerate(self.weights):
            # calculate net inputs
            net_inputs = np.dot(activations, w) # if two matrcies: matrix multiplicaiton; if two vectors, dot product

            # calculate activations 
            activations = self.sigmoid(net_inputs)
            self.activations[i+1] = activations

        return activations
    
    
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    

if __name__ == "__main__":

    # create a dataset to train a network for the sum operation
    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])

    # create an MLP
    mlp = MLP(2, [5], 1)

    # train our mlp
    mlp.train(inputs, targets, 50, 0.1)

    # create dummy set
    input = np.array([0.3, 0.1])
    target = np.array([0.4])

    output = mlp.forward_propagate(input)
    print()
    print()
    print(f"Our network believes that {input[0]} + {input[1]} is equal to {output[0]}")