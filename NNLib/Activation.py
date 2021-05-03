#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def rbf(x):
    return np.exp(-x**2)
    
def rbf_deriv(x):
    return -2*x*rbf(x)

def sigmoid(x):
    """ Sigmoid activation function """
    output = np.zeros_like(x)
    output[x>=0] = 1.0/(1.0+np.exp(-x[x>=0])) # handles overflow for -inf
    z = np.exp(x[x<0])
    output[x<0] = z/(1+z) # handles overflow for +inf
    return output
    
def tanh(x):
    return 2*sigmoid(2*x)-1

def tanh_deriv(x):
    return (1-(tanh(x))**2)

def sigmoid_deriv(x):
    """ Derivative of the sigmoid function """
    return sigmoid(x)*(1-sigmoid(x))

def softmax(x):
    """ Softmax activation function\n
    May return 0 probabilities. """
    x_maxed = x - np.max(x,axis=0)
    exp_x = np.exp(x_maxed) # By subtracting max it adds numerical stability
    return exp_x/np.sum(exp_x, axis=0)

def relu(x):
    """ ReLU activation function """
    y = np.copy(x)
    y[y<0] = 0
    return y

def relu_deriv(x):
    return step(x)

def linear(x):
    """ Linear activation function """
    return np.copy(x)

def linear_deriv(x):
    return np.ones(x.shape)

def step(x):
    y = np.zeros(x.shape)
    y[x > 0] = 1
    return y

def get_act_fcn_deriv(act_fcn):
    if act_fcn == relu:
        act_fcn_deriv = relu_deriv
    elif act_fcn == linear:
        act_fcn_deriv = linear_deriv
    elif act_fcn == sigmoid:
        act_fcn_deriv = sigmoid_deriv
    elif act_fcn == rbf:
        act_fcn_deriv = rbf_deriv
    elif act_fcn == tanh:
        act_fcn_deriv = tanh_deriv
    else:
        print("Warning: else clause of get_act_fcn_deriv()")
        print("Softmax must use cross entropy loss")
        act_fcn_deriv = lambda x: 1
    return act_fcn_deriv
