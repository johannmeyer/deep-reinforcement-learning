#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np # general math library
import copy # for deepcopy of model parameters
import pickle # for saving model
from NNLib import Activation # to get activation function derivatives
from NNLib import Initialiser # for default weight initialisation

class NeuralNetwork:
    
    def __init__(self, dims, act_fcns, init_fcn=Initialiser.glorot_uniform_init):
        """ Initialises neural network.\n
        Generates the layers, the random weights and sets some default
        parameters. """
        self.num_layers = len(dims)
        
        assert self.num_layers-1 == len(act_fcns), "Check number of activations is equal to number of layers -1"
        
        # randomly initialise weights
        self.dims = dims
        self.weights, self.biases = init_fcn(dims)
        self.act_fcn = act_fcns
        
        # store the derivatives of the activation functions in a list for backprop
        self.act_fcn_deriv = []
        for act_fcn in self.act_fcn:
            self.act_fcn_deriv.append(Activation.get_act_fcn_deriv(act_fcn))
        
        self.reg = 0 # regularisation weight (aka weight decay)
    
    def predict(self, x, weights=None, biases=None):
        """ Performs a forward pass in the neural network given input x and
        stores the activations for the backward pass. """
        if weights is None or biases is None:
            weights = self.weights
            biases = self.biases
        activation = x
        self.activations = [x]
        self.zs = [] # stores all the ouputs before activations
        for (w, b, act_fcn) in zip(weights, biases, self.act_fcn):
            z = (w @ activation) + b
            self.zs.append(z)
            activation = act_fcn(z)
            self.activations.append(activation)
            
        return activation
    
    def parameter_gradient(self, weights=None, biases=None, init_gradient=None, act_loss_merged=True):
        """ Implements backpropagation.\n
        Note this function should not be called externally as it requires a 
        forward pass to be performed first. Forward pass is not done explicitly
        since Reinforcement Learning and specifically Policy Gradient methods
        usually need to apply a scaling to the loss function. """
        if weights is None or biases is None:
            weights = self.weights
            biases = self.biases
        
        num_samples_per_batch = self.zs[-1].shape[1]
        
        if init_gradient is None:
            delta = 1/num_samples_per_batch
        else:
            delta = init_gradient/num_samples_per_batch

        if not act_loss_merged:
            delta = delta*self.act_fcn_deriv[-1](self.zs[-1])
        
        
        nabla_w = [np.zeros(w.shape) for w in weights] # Stores weight gradients
        nabla_b = [np.zeros(b.shape) for b in biases] # Stores bias gradients
        
        nabla_w[-1] = delta @ self.activations[-2].transpose() + self.reg*self.weights[-1]
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)
        
        for l in range(2, self.num_layers):
            z = self.zs[-l]
            delta = (weights[-l+1].transpose() @ delta) * self.act_fcn_deriv[-l](z)
            nabla_w[-l] = delta @ self.activations[-l-1].transpose() + self.reg*self.weights[-l]
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)
        
        return (nabla_w, nabla_b)
    
    def input_gradient(self, x):
        """ Computes the gradient of the output of the neural network with
        respect to the input of the neural network. """
        
        # forward pass
        self.predict(x)
        
        # Backward pass
        delta = self.act_fcn_deriv[-1](self.zs[-1])
        for l in range(2, self.num_layers):
            z = self.zs[-l]
            delta = (self.weights[-l+1].transpose() @ delta) * self.act_fcn_deriv[-l](z)
        
        nabla_input = self.weights[-l].transpose() @ delta
        return nabla_input
    
    def get_weights(self):
        return self.weights, self.biases
#        return (copy.deepcopy(self.weights), copy.deepcopy(self.biases))
    
    def set_weights(self, weights, biases):
        # use deepcopy to avoid target_model and normal model from using 
        # the same weights. (standard copy means object references instead of 
        # values are copied)
        self.weights = copy.deepcopy(weights)
        self.biases = copy.deepcopy(biases)
        
    def blend_weights(self, target_network, tau=0.1):
        self.weights = [tau*target_weights + (1-tau)*weights for weights, target_weights in zip(self.weights, target_network.weights)]
        self.biases = [tau*target_biases + (1-tau)*biases for biases, target_biases in zip(self.biases, target_network.biases)]
    
    def save_weights(self, name):
        pickle.dump([self.weights, self.biases], open("{}.pickle".format(name), "wb"))
        
    def load_weights(self, name):
        try:
            self.weights, self.biases = pickle.load(open("{}.pickle".format(name), "rb"))
        except:
            print("Could not load weights: File Not Found")
        
    
    
