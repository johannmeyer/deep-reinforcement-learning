#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # used for 3D plots

from NNLib import Activation, Loss, Optimiser, Initialiser
from NNLib.NeuralNetwork import NeuralNetwork

if __name__ == "__main__":
    np.random.seed(0)
    plt.close('all')
    # Create the neural net
    model = NeuralNetwork([2,30,1], [Activation.rbf, Activation.linear])
    model.reg = 1e-3
    optimiser = Optimiser.Momentum(model, loss_fcn=Loss.mean_squared_error, nesterov=True, batch_size=32)

    # Create data grid
    x = np.arange(0,2,0.1)
    y = np.arange(0,2,0.1)
    x, y = np.meshgrid(x,y)
    
    # Convert the meshgrid into column vectors
    x_orig_shape = x.shape
    x = x.reshape(1,-1)
    y = y.reshape(1,-1)
    
    num_samples = x.shape[1]
    
    XY = np.concatenate((x,y),axis=0)
    z = 0.2*np.exp(x**2)*np.sin(10*x)-0.5*np.exp(x)+np.cos(5*y) #+ 0.01*np.random.randn(1, num_samples)
    z_noisy = z# +np.random.randn(*z.shape)
    
    # Initial loss
    z_init = model.predict(XY)
    
    # Train neural net
    plt.figure()
    losses = np.zeros((30,2))           
    learning_rate = 1e-4
    for i in range(30):
        optimiser.train(XY, z_noisy, num_epochs=30, learning_rate=learning_rate)
        z_predict = model.predict(XY)
        loss = Loss.mean_squared_error(z, z_predict)
        losses[i,:] = i*30, loss
        learning_rate *= 0.95
    
    plt.semilogy(losses[:,0], losses[:,1], '-')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    # Plot 3D predictions vs reference data
    z_predict = z_predict.reshape(*x_orig_shape)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x.reshape(*x_orig_shape), y.reshape(*x_orig_shape), z_init.reshape(*x_orig_shape), color='g', label='Initial Net')
    ax.plot_wireframe(x.reshape(*x_orig_shape), y.reshape(*x_orig_shape), z_predict, color='b', label='Trained Net')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
#    ax.plot_wireframe(x.reshape(*x_orig_shape), y.reshape(*x_orig_shape), z_noisy.reshape(*x_orig_shape), color='g', label='Dataset Noisy')
    ax.plot_wireframe(x.reshape(*x_orig_shape), y.reshape(*x_orig_shape), z.reshape(*x_orig_shape), color='r', label='Dataset')
    ax.plot_wireframe(x.reshape(*x_orig_shape), y.reshape(*x_orig_shape), z_predict, color='b', label='Trained Net')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    
    
    print(np.linalg.norm(model.weights[0]))
    plt.show()
