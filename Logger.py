#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import time
from collections import deque
import sys
import os

from IPython import embed


class Logger:
    """ This class creates a method to log some value to a file and generates a
    x axis with values from 1 - N, where N is the number of values in the file.
    It can, for example, be used to keep track of the mean reward of an agent in
    an environment. """
    
    def __init__(self, log_name='log.txt', reset=False, append_date=True, log_dir='./'):
        date_str =  '_' + time.strftime('%m-%d_%H-%M-%S') if append_date else ''
        self.log_name = log_dir + log_name + date_str
        
        self.x_data = []
        self.y_data = []
        
    
    def append(self, values):
        if self.x_data != []:
            last_episode_num = self.x_data[-1]
            self.x_data.append(np.arange(last_episode_num+1, last_episode_num + len(values)+1))
        else:
            self.x_data.append(np.arange(1, len(values)+1))
            
        self.y_data.append(values)
    
    def save(self, data=None, log_name=None):
        """ Saves the y-data of the logger or, alternatively, the specified data. """
        if data is None:
            data = self.y_data
        if log_name is None:
            log_name = self.log_name
            
        np.save(log_name, data)
            
    def load(self, log_name=None):
        """ Loads the data given by the provided log name or, alternatively, the
        log name used when creating the Logger. """
        if log_name is None:
            log_name = self.log_name
        self.y_data = np.load(log_name)
        num_data_pts = len(self.y_data)
        self.x_data = np.arange(1, num_data_pts +1)
    
    def plot(self):
        """ Used to plot the mean reward stored in the Logger object. """        
        plt.figure()
        plt.plot(self.x_data, self.y_data, '-')
        plt.ylabel('Mean Reward [-]')
        plt.xlabel('Episode Number [-]')
        plt.show()
        
def moving_average(data, maxlen=100):
    mean_reward_deque = deque(maxlen=maxlen)
    
    smooth_data = []
    
    for y in data:
        mean_reward_deque.append(y)
        smooth_data.append(np.mean(mean_reward_deque))
    return smooth_data

def tsplot(ax, data, ci=2,**kw):
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    #sd = np.std(data, axis=0)
    #cis = (est - ci*sd, est + ci*sd)
    cis = (np.min(data,axis=0), np.max(data,axis=0))
    ax.fill_between(x,cis[0],cis[1],alpha=0.2)
    ax.plot(x,est,**kw)
    ax.margins(x=0)
        
def plot_dir(ax, folder_name, root_folder="./"):
    folder_data = []
    folder_path = root_folder + folder_name
    for file_name in os.listdir(folder_path):
        logger = Logger()
        logger.load('%s/%s' %(folder_path, file_name))
        folder_data.append(moving_average(logger.y_data))

    tsplot(ax, np.array(folder_data), label=folder_name)

    
if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Not enough arguments: missing folder names')
        sys.exit(0)
    
    plt.figure()
    ax = plt.gca()
    
    if sys.argv[1] == '-d':
        folder_name = sys.argv[2]
        for file_name in os.listdir(folder_name):
            logger = Logger()
            logger.load('%s/%s' %(folder_name, file_name))
            print(file_name, np.max(logger.y_data))
            plt.plot(moving_average(logger.y_data),label=file_name[-12:-4])  
        ax.legend()
        plt.show()
    elif sys.argv[1] == '-f':
        file_name = sys.argv[2]
        logger = Logger()
        logger.load(file_name)
        plt.plot(logger.x_data, moving_average(logger.y_data))
        plt.show()
    else:
        labels = []
        root_folder = sys.argv[1]
        for folder_name in os.listdir(root_folder):
            plot_dir(ax, folder_name, root_folder)
        
        ax.legend()
        ax.set_xlabel('Episode [-]')
        ax.set_ylabel('Mean Reward [-]')
        plt.grid(b=True, which='minor', color='0.80', linestyle='-')
        plt.minorticks_on()
        
        plt.pause(0.001)
        plt.pause(0.001)
        plt.ion()
        embed()
    
