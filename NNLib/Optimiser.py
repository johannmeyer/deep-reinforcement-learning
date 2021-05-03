#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from NNLib import Loss

class Optimiser:
    
    def __init__(self, neural_net, learning_rate=1e-3, loss_fcn=None, act_loss_merged=True):
        self.learning_rate = learning_rate
        self.neural_net = neural_net
        self.loss_fcn = loss_fcn
        if self.loss_fcn is not None:
            self.loss_fcn_deriv = Loss.get_loss_fcn_deriv(self.loss_fcn)
            self.act_loss_merged = act_loss_merged
        else:
            self.act_loss_merged = False

class Adam(Optimiser):
    
    def __init__(self, neural_net, learning_rate=1e-3, loss_fcn=None, B1=0.9, B2=0.999, eps=10e-8, batch_size=None):
        
        super().__init__(neural_net, learning_rate, loss_fcn)
        
        self.B1 = B1
        self.B2 = B2
        self.eps = eps
        
        self.v_weights = [np.zeros(w.shape) for w in neural_net.weights] # Stores weight momentum
        self.v_bias = [np.zeros(b.shape) for b in neural_net.biases] # Stores bias momentum
        self.m_weights = [np.zeros(w.shape) for w in neural_net.weights] # Stores weight momentum
        self.m_bias = [np.zeros(b.shape) for b in neural_net.biases] # Stores bias momentum
        self.t = 0
        
        self.batch_size = batch_size
        
    def train(self, x, target, num_epochs=1, init_gradient=None, learning_rate=None):
        if learning_rate is None:
            learning_rate = self.learning_rate
        
        num_samples = x.shape[1]
        
        for epoch in range(num_epochs):
            
            batch_ind = get_batch(num_samples, self.batch_size, stochastic=True)
            
            for batch_i in batch_ind:
                self.t += 1
                
                # extract batch
                x_i = x[:,batch_i]
                batch_estimate = self.neural_net.predict(x_i)
                
                if init_gradient is not None:
                    batch_init_gradient = init_gradient[:,batch_i]
                else:
                    batch_init_gradient = 1
                if self.loss_fcn is not None:
                    if self.loss_fcn is not Loss.cross_entropy_loss:
                        batch_target = target[:,batch_i]
                    else:
                        batch_target = target[batch_i]
                    batch_init_gradient = batch_init_gradient*self.loss_fcn_deriv(batch_target, batch_estimate)
                
                # Backpropagate errors and compute gradients
                (nabla_w, nabla_b) = self.neural_net.parameter_gradient(init_gradient=batch_init_gradient, act_loss_merged=self.act_loss_merged)
#                print(nabla_w)
#                sys.exit(0)
#                new_nabla_w = []
#                for dw in nabla_w:
#                    norm = np.linalg.norm(dw)
#                    if norm > 0.1:
#                        dw = dw/norm*0.1
#                    new_nabla_w.append(dw)
#                nabla_w = new_nabla_w
                
#                new_nabla_b = []
#                for db in nabla_b:
#                    norm = np.linalg.norm(db)
#                    if norm > 0.1:
#                        db = db/norm*0.1
#                    new_nabla_b.append(db)
#                nabla_b = new_nabla_b
#                nabla_w = [dw/np.linalg.norm(dw) if np.linalg.norm(dw) > 1 else dw for dw in nabla_w]
#                nabla_b = [db/np.linalg.norm(db) if np.linalg.norm(db) > 1 else db for db in nabla_b]
                
                # correct with momentum
                self.m_weights = [self.B1*m + (1-self.B1)*dw for m, dw in zip(self.m_weights, nabla_w)]
                self.m_bias =    [self.B1*m + (1-self.B1)*db for m, db in zip(self.m_bias, nabla_b)]
                
                self.v_weights = [self.B2*v + (1-self.B2)*dw**2 for v, dw in zip(self.v_weights, nabla_w)]
                self.v_bias = [self.B2*v + (1-self.B2)*(db)**2 for v, db in zip(self.v_bias, nabla_b)]
                
                # Compute learning rate
                alpha_t = learning_rate*np.sqrt(1-self.B2**self.t)/(1-self.B1**self.t)
                
                # Update weights
                self.neural_net.biases  = [b-alpha_t*m/(np.sqrt(v) + self.eps) for b, m, v in zip(self.neural_net.biases, self.m_bias, self.v_bias)]
                self.neural_net.weights = [w-alpha_t*m/(np.sqrt(v) + self.eps) for w, m, v in zip(self.neural_net.weights, self.m_weights, self.v_weights)]

class GradientDescent(Optimiser):
    
    def __init__(self, neural_net, learning_rate=1e-3, loss_fcn=None, variant='batch_GD'):
        super().__init__(neural_net, learning_rate, loss_fcn)
        
        if variant == 'batch_GD':
            self.optimiser = self.batch_GD
        elif variant == 'SGD':
            self.optimiser = self.SGD
        elif variant == 'batch_SGD':
            self.optimiser = self.batch_SGD
        else:
            print('Error: Invalid Varaint')
        
    def train(self, x, target, num_epochs=1, init_gradient=None, learning_rate=None):
        self.optimiser(x, target, num_epochs=1, init_gradient=None, learning_rate=learning_rate)
        
    def batch_GD(self, x, target, num_epochs=1, init_gradient=None, learning_rate=None):
        """ Batch Gradient Descent (Vanilla Gradient Descent).\n
        Performs one smoothed weight update per epoch. Learning is much
        smoother than Stochastic Gradient Descent. Batch size is always equal
        to number of samples."""
        
        for epoch in range(num_epochs):
            self.gradient_descent_update(x, target, init_gradient, learning_rate=learning_rate)
    
    def SGD(self, x, target, num_epochs=1, init_gradient=None, learning_rate=None):       
        """ Stochastic Gradient Descent.\n
        This algorithm performs a weight update for each input. In other words,
        it performs 'num_samples' updates per epoch. Learning is faster than 
        Batch Gradient Descent but also more erratic."""
        num_samples = x.shape[1]
        
        for epoch in range(num_epochs):
            # Uncorrelate updates
            ind = np.random.permutation(num_samples)
            for i in ind:
                # i:i+1 ensures x_i the array does not remove a dimension
                if init_gradient is not None:
                    batch_init_gradient = init_gradient[:,i:i+1]
                else:
                    batch_init_gradient = None
                x_i = x[:,i:i+1]
                target_i = target[:,i:i+1]
                self.gradient_descent_update(x_i, target_i, batch_init_gradient, learning_rate=learning_rate)
    
    def batch_SGD(self, x, target, num_epochs=1, init_gradient=None, learning_rate=None):
        """ Batch Stochastic Gradient Descent.\n
        This is stochastic gradient descent using batch sizes larger than 1.
        The benefit is that learning is much smoother at the cost of fewer
        updates per epoch. """
        num_samples = x.shape[1]
        
        
        for epoch in range(num_epochs):
            # Uncorrelate updates
            batch_ind = get_batch(num_samples, self.batch_size, stochastic=True)

            for batch_i in batch_ind:
                # extract batch
                if init_gradient is not None:
                    batch_init_gradient = init_gradient[:,batch_i]
                else:
                    batch_init_gradient = None
                x_i = x[:,batch_i]
                target_i = target[:,batch_i]
                self.gradient_descent_update(x_i, target_i, batch_init_gradient, learning_rate=learning_rate)
        
    def gradient_descent_update(self, batch_x, batch_target, init_gradient=None, learning_rate=None):
        if learning_rate is None:
            learning_rate = self.learning_rate
            
        batch_estimate = self.neural_net.predict(batch_x)
        # Backpropagate errors and compute gradients
        if init_gradient is None:
            batch_init_gradient = 1
        if self.loss_fcn is not None:
            batch_init_gradient *= self.loss_fcn_deriv(batch_target, batch_estimate)
            
        corrections = self.neural_net.parameter_gradient(init_gradient=batch_init_gradient, act_loss_merged=self.act_loss_merged)
        
        # Update weights
        (self.neural_net.weights, self.neural_net.biases) = update_weights(self.neural_net.get_weights(), corrections, learning_rate)
            
class Momentum(Optimiser):
    def __init__(self, neural_net, learning_rate=1e-3, loss_fcn=None, gamma=0.9, nesterov=True, batch_size=None):
        super().__init__(neural_net, learning_rate, loss_fcn)
        
        self.gamma = 0.9
        self.v_weights = [np.zeros(w.shape) for w in self.neural_net.weights] # Stores weight momentum
        self.v_bias = [np.zeros(b.shape) for b in self.neural_net.biases] # Stores bias momentum
        self.nesterov = nesterov
        
        self.batch_size = batch_size
        
    def train(self, x, target, num_epochs=1, init_gradient=None, learning_rate=None):
        if learning_rate is None:
            learning_rate = self.learning_rate
        
        num_samples = x.shape[1]
        
        for epoch in range(num_epochs):
            # Get indices for each batch
            batch_ind = get_batch(num_samples, self.batch_size, stochastic=True)
#            print('epoch', epoch)
            for batch_i in batch_ind:
#                print ('batch_i', batch_i)
                # extract batch
                batch_x = x[:,batch_i]
                
#                print('train x',batch_x)
#                print('train y', batch_target)
                if self.nesterov:
                    weight_estimate, bias_estimate = update_weights(self.neural_net.get_weights(), (self.v_weights, self.v_bias), learning_rate=self.gamma)
                else:
                    weight_estimate, bias_estimate = self.neural_net.get_weights()
                    
                batch_estimate = self.neural_net.predict(batch_x, weight_estimate, bias_estimate)
                
                if init_gradient is not None:
                    batch_init_gradient = init_gradient[:,batch_i]
#                    print('init grad', batch_init_gradient)
                else:
                    batch_init_gradient = 1
                if self.loss_fcn is not None:
                    if self.loss_fcn is not Loss.cross_entropy_loss:
                        batch_target = target[:,batch_i]
                    else:
                        batch_target = target[batch_i]
                    batch_init_gradient = batch_init_gradient*self.loss_fcn_deriv(batch_target, batch_estimate)
                    
                # Backpropagate errors and compute gradients
#                assert batch_init_gradient == self.loss_fcn_deriv(batch_target, batch_estimate)
                (nabla_w, nabla_b) = self.neural_net.parameter_gradient(weight_estimate, bias_estimate, init_gradient=batch_init_gradient, act_loss_merged=self.act_loss_merged)
                
#                nabla_w = [dw/np.linalg.norm(dw) if np.linalg.norm(dw) > 1 else dw for dw in nabla_w]
#                nabla_b = [db/np.linalg.norm(db) if np.linalg.norm(db) > 1 else db for db in nabla_b]
#                new_nabla_w = []
#                for dw in nabla_w:
#                    norm = np.linalg.norm(dw)
#                    if norm > 0.1:
#                        dw = dw/norm*0.1
#                    new_nabla_w.append(dw)
#                nabla_w = new_nabla_w
#                
#                new_nabla_b = []
#                for db in nabla_b:
#                    norm = np.linalg.norm(db)
#                    if norm > 0.1:
#                        db = db/norm*0.1
#                    new_nabla_b.append(db)
#                nabla_b = new_nabla_b
                
                # correct with momentum
                self.v_weights = [self.gamma*v + learning_rate*dw for v, dw in zip(self.v_weights, nabla_w)]
                self.v_bias =    [self.gamma*v + learning_rate*db for v, db in zip(self.v_bias, nabla_b)]
                
                # Update weights
                (self.neural_net.weights, self.neural_net.biases) = update_weights(self.neural_net.get_weights(), (self.v_weights, self.v_bias), learning_rate=1)
        
def get_batch(num_samples, batch_size, stochastic=False):
    """ Generates a list of indices for each batch. """
    if batch_size == None:
        batch_size = num_samples

    num_batches = num_samples//batch_size
    # if num_samples is not perfectly divisible by the batch_size increment
    # num_batches by one to make all but the last batch equal to batch_size
    if num_samples % batch_size != 0:
        num_batches += 1
        
    # Creates a list of arrays to index the random indices for each batch
    # Desired batch size not guanranteed equal to actual batch size
    if stochastic:
        batch_ind = np.array_split(np.random.permutation(num_samples), num_batches)
    else:
        batch_ind = np.array_split(np.arange(num_samples), num_batches)
    return batch_ind

def update_weights(parameters, corrections, learning_rate):
    weights, biases = parameters
    weight_correction, bias_correction = corrections
    
#    weight_correction = [dw/np.linalg.norm(dw) if np.linalg.norm(dw) > 1 else dw for dw in weight_correction]
#    bias_correction = [db/np.linalg.norm(db) if np.linalg.norm(db) > 1 else db for db in bias_correction]
    
    biases  = [b-learning_rate*db for b, db in zip(biases, bias_correction)]
    weights = [w-learning_rate*dw for w, dw in zip(weights, weight_correction)]
    return (weights, biases)
