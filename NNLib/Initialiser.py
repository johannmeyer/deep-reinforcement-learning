#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def glorot_uniform_init(dims):
    weights = [np.sqrt(6/(x+y))*(np.random.rand(y, x)*2-1) for x, y in zip(dims, dims[1:])]
    biases =  [np.zeros((y,1)) for y in dims[1:]]
    return (weights, biases)

def rand_normal_init(dims, mean, variance):
    weights = [np.random.randn(y, x)*np.sqrt(variance)+mean for x, y in zip(dims, dims[1:])]
    biases =  [np.random.randn(y,1)*np.sqrt(variance)+mean for y in dims[1:]]
    return (weights, biases)

def rand_uniform_init(dims, limit=1):
    u_range = 2*limit
    weights = [np.random.rand(y, x)*u_range - limit for x, y in zip(dims, dims[1:])]
    biases =  [np.random.rand(y,1)*u_range - limit for y in dims[1:]]
    return (weights, biases)

def xavier_init(dims):
    weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(dims, dims[1:])]
    biases = [np.zeros((y, 1)) for x, y in zip(dims, dims[1:])]
    return (weights, biases)
