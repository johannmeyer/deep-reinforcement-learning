#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def mean_squared_error(target, estimate):
    # Average loss per sample
    num_samples = target.shape[1]
    return 0.5*np.sum((target-estimate)**2)/num_samples

def mean_squared_error_deriv(target, estimate):
#    num_samples = target.shape[1]
    return (estimate - target)

def cross_entropy_loss(target, estimate):
    """ Computes the average cross entropy loss for a batch of data"""
    # Target is the indices of the correct label in estimate
    num_samples = target.shape[0]
    return np.sum(-np.log(estimate[target, range(num_samples)]))/num_samples

def cross_entropy_loss_deriv(target, estimate):
    """ Computes the derivative of the log loss of the cross entropy for the
    softmax function. """
    num_samples = target.shape[0]
    gradient = estimate.copy()
    gradient[target, range(num_samples)] -= 1
    return gradient

def get_loss_fcn_deriv(loss_fcn):
    if loss_fcn == mean_squared_error:
        loss_fcn_deriv = mean_squared_error_deriv
    elif loss_fcn == cross_entropy_loss:
        loss_fcn_deriv = cross_entropy_loss_deriv
    else:
        print("Warning: loss function derivative not defined!")
    return loss_fcn_deriv

def loss(model, loss_fcn,  x, target):
        """ Compute the loss of the last prediction """
        # get estimate from NN
        estimate = model.predict(x)
        return loss_fcn(target, estimate)
