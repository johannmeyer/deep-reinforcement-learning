#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import signal
import sys

import argparse

from Algorithms import ddpg, REINFORCE, A2C, GAE, TD3
from Logger import Logger
from collections import deque

import numpy as np

class Handler:
    def __init__(self, args):
        args.det = False
        
        self.args = args
        
        signal.signal(signal.SIGINT, self.handler)

    def handler(self, signal, frame):
        key = input("Press 'q' to exit or 't' to toggle rendering or 'd' for deterministic or 's' to save models or 'm' to set new number of episodes: ")
        if key == 't':
            self.args.render = not self.args.render
        elif key == 'q':
            sys.exit()
        elif key == 'd':
            self.args.det = not self.args.det
        elif key == 's':
            for model in self.args.models:
                file_name = input("Enter model %s weights file name: " %(type(model)))
                model.save_weights(file_name)
        elif key == 'g':
            self.args.save_render = not self.args.save_render
        elif key == 'm':
            try:
                self.args.num_episodes = int(input('New number of episodes: '))
            except:
                pass

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm', type=str, help='Specify which algorithm to use {REINFORCE, A2C, GAE, DDPG, DQN}')
    parser.add_argument('env_name', type=str, help='Discrete Action Space = {CartPole-v1, LunarLander-v2} Continuous Action Space = {Pendulum-v0, LunarLanderContinuous-v2, BipedalWalker-v2}')

    parser.add_argument('--log_dir', help='Add logging at specified directory location')
    parser.add_argument('--render', default=False, help='Enable rendering')
    
    parser.add_argument('--gamma', default=0.99, type=float, help='Discounting factor')
    parser.add_argument('--Lambda', default=1, type=float, help='Generalised Advantage Estimation lambda discounting factor')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, help='Main learning rate for algorithm or actor learning rate')
    parser.add_argument('--lr_c', '--learning_rate_critic', default=1e-4, type=float, help='Learning rate for critic')
    parser.add_argument('--baseline', action='store_const', const=True, help='Only used if algorithm==REINFORCE and enables a state-dependent baseline to reduce variance')
    parser.add_argument('--batch_size', default=32, type=int, help='The size of the batches that the algorithm will train on')
    parser.add_argument('--episode_batch_size', default=1, type=int, help='Only used in REINFORCE and determines the number of episodes to use for smoothing the gradient')
    parser.add_argument('--replay_mem_size', default=1e6, type=float, help='Size of the replay memory')
    parser.add_argument('--tau', default=1e-3, type=float, help='Polyak constant for blending the target and active models')
    parser.add_argument('--hidden_layers', default=[20], nargs='*', type=int, help='Dimensions of each hidden layer')
    
    parser.add_argument('--num_episodes', default=1000, type=int, help='Number of episodes to train algorithm')
    
    args = parser.parse_args()
    
    # For monitoring 100-episode reward average
    args.mean_reward_deque = deque(maxlen=100)
    
    args.save_render = False
    
    handler = Handler(args)
    if args.algorithm == 'DDPG':
        if args.log_dir is not None:
            args.reward_logger = Logger('%s_%s_%d_%f_lra-%f_lrc-%f_%f' %(args.algorithm, args.env_name, args.batch_size, args.gamma, args.lr, args.lr_c, args.replay_mem_size), log_dir=args.log_dir)
        ddpg.DDPG(args)
    elif args.algorithm == 'TD3':
        if args.log_dir is not None:
            args.reward_logger = Logger('%s_%s_%d_%f_lra-%f_lrc-%f_%f' %(args.algorithm, args.env_name, args.batch_size, args.gamma, args.lr, args.lr_c, args.replay_mem_size), log_dir=args.log_dir)
        TD3.TD3(args)
    elif args.algorithm == 'REINFORCE':
        if args.log_dir is not None:
            if args.baseline:
                args.algorithm += '-baseline'
                args.reward_logger = Logger('%s_%s_%f_lr_p-%f_lr_v-%f' %(args.algorithm, args.env_name, args.gamma, args.lr, args.lr_c), log_dir=args.log_dir)
            else:
                args.reward_logger = Logger('%s_%s_%f_lr_p-%f' %(args.algorithm, args.env_name, args.gamma, args.lr), log_dir=args.log_dir)
        REINFORCE.REINFORCE(args)
    elif args.algorithm == 'A2C':
        if args.log_dir is not None:
            args.reward_logger = Logger('%s_%s_%d_%f_lra-%f_lrc-%f' %(args.algorithm, args.env_name, args.batch_size, args.gamma, args.lr, args.lr_c), log_dir=args.log_dir)
        A2C.A2C(args)
    elif args.algorithm == 'GAE':
        if args.log_dir is not None:
            args.reward_logger = Logger('%s_%s_%d_%f_lra-%f_lrc-%f_lambda-%f' %(args.algorithm, args.env_name, args.batch_size, args.gamma, args.lr, args.lr_c, args.Lambda), log_dir=args.log_dir)
        GAE.GAE(args)
