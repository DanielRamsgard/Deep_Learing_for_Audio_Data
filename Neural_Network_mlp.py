import numpy as np


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


    def forward_propagate(self, inputs):
        activations = inputs
        for w in self.weights:
            # calculate net inputs
            net_inputs = np.dot(activations, w) # if two matrcies: matrix multiplicaiton; if two vectors, dot product

            # calculate activations 
            activations = self.sigmoid(net_inputs)

        return activations
    
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    

if __name__ == "__main__":

    # create an MLP
    mlp = MLP()

    # create some inputs
    inputs = np.random.rand(mlp.num_inputs)

    # perform forward prop
    outputs = mlp.forward_propagate(inputs)

    # print results
    print(f"The network input is: {inputs}")
    print(f"The network output is: {outputs}")