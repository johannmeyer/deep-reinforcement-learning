#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import deque
import random
import numpy as np

class ReplayMemory:
    def __init__(self, max_size):
        self.max_size = int(max_size)
        self.replay_memory = deque(maxlen=self.max_size)
        
    def get_batch(self, batch_size):
        minibatch = random.sample(self.replay_memory, min(batch_size, len(self.replay_memory)))
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        # Convert from tuples to neural network compatible inputs
        states = np.array(states).T
        actions = np.array(actions).T
        rewards = np.array(rewards)
        next_states = np.array(next_states).T
        dones = np.array(dones)
        return states, actions, rewards, next_states, dones
    
    def append(self, transition):
        self.replay_memory.append(transition)
        
    def insert(self, transition, location):
        self.replay_memory[location] = transition
