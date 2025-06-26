import random
import numpy as np

class Network:
    def __init__(self, sizes:list):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


    def feed_foreward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a
    


    def train_model(self, training_data, epochs, mini_batch_size, learning_rate, validation_data):

        validation_data_length = len(validation_data)
        training_data_length = len(training_data)

        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, training_data_length, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
                # 
                # 
                # 
                # 


    def update_mini_batch(self, mini_batch, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # print("nabla_b", nabla_b)
        # print("nabla_w", nabla_w)

        for x, y in mini_batch: # This means for each training example in the current mini_batch. 
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]

        # self.weights = [w - (learning_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        # self.biases = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        # Remember here that x is the 784 x 1 pixle matrix and y is the associated 1 x 15 label. 
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Before we can do the backward pass we need to keep track of all the activation and preactivation values so that we can actually do a backward pass. 
        # We will do this by recording their values as we do a foward pass. 
        activation = x
        activations = [x] # *** Notes 0.1 *** This is a list to store all of the activations, layer by layer.
        zs = [] # This is a list to store all the z vectors layer by layer. 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        loss = 0.5 * np.sum((activations[-1] - y) ** 2) # *** Notes 0.2 ***
        
        # Now that we have done the forward pass and have the loss we need to do backpropagation. 
        # Start by finding the partial derivative of the loss w.r.t the final layer neurons post-activation. 
        delta = activations[-1] - y

        # Now that we have the first part of our 
        for l in range(-1, -self.num_layers, -1):
            z_delta = sigmoid_prime(delta) 
            delta = 

                







def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
























