#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from NNLib import Activation, Loss, Optimiser
from NNLib.NeuralNetwork import NeuralNetwork

import numpy as np

class Actor:
    def __init__(self, num_states, num_actions, env_action_limit=(-1,1), learning_rate=1e-4, hidden_layers=[20]):
        self.num_states = num_states
        self.num_actions = num_actions
        
        self.env_action_min = env_action_limit[0]
        self.env_action_max = env_action_limit[1]
        assert np.all(-self.env_action_min == self.env_action_max)
        self.env_action_max = np.expand_dims(self.env_action_max, axis=1)
        
        self.model = self.create_model(num_states, num_actions, hidden_layers)
        self.target_model = self.create_model(num_states, num_actions, hidden_layers)
        self.target_model.set_weights(*self.model.get_weights())
        self.optimiser = Optimiser.Adam(self.model, learning_rate)
        
    def create_model(self, num_states, num_actions, hidden_layers):
        model = NeuralNetwork([num_states, *hidden_layers, num_actions],
                              [*len(hidden_layers)*[Activation.relu], Activation.tanh])
        model.reg = 1e-4
        return model
    
    def predict(self, state, target=False):
        if target:
            model = self.target_model
        else:
            model = self.model
        actions = model.predict(state)
        actions = self.env_action_max * actions
        return actions

    
    def train(self, states, action_gradient):
        num_samples = states.shape[1]
        self.optimiser.train(states, np.zeros((self.num_actions, num_samples)), init_gradient= - self.env_action_max*action_gradient)

    def blend_weights(self, tau=1e-3):
        self.target_model.blend_weights(self.model, tau)



class Critic:
    def __init__(self, num_states, num_actions, learning_rate=1e-3, hidden_layers=[20]):
        self.num_states = num_states
        self.num_actions = num_actions
        
        self.model = self.create_model(num_states, num_actions, hidden_layers)
        self.target_model = self.create_model(num_states, num_actions, hidden_layers)
        self.target_model.set_weights(*self.model.get_weights())
        
        self.optimiser = Optimiser.Adam(self.model, learning_rate, loss_fcn=Loss.mean_squared_error)
    
    def predict(self, state, action, target=False):
        if target:
            model = self.target_model
        else:
            model = self.model
        return model.predict(np.vstack((state, action)))
    
    def create_model(self, num_states, num_actions, hidden_layers):
        model = NeuralNetwork([num_states + num_actions, *hidden_layers, 1],
                              [*len(hidden_layers)*[Activation.relu], Activation.linear])
        model.reg = 1e-4
        
        return model
    
    def train(self, state, action, target):
        self.optimiser.train(np.vstack((state, action)), target)
    
    def get_action_gradient(self, state, action):
        return self.model.input_gradient(np.vstack((state, action)))[self.num_states:]

    def blend_weights(self, tau=1e-3):
        self.target_model.blend_weights(self.model, tau)
