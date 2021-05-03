#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from NNLib import Activation, Loss, Optimiser, Initialiser
from NNLib.NeuralNetwork import NeuralNetwork

import gym
import numpy as np

def discount_rewards(rewards, gamma):
    cum_reward = 0
    discounted_rewards = np.empty_like(rewards, dtype=float)
    for i, reward in enumerate(reversed(rewards)):
        cum_reward *= gamma
        cum_reward += reward
        discounted_rewards[-i-1] = cum_reward
    return discounted_rewards 

def REINFORCE(args):
    
    # Initialise Environment
    env = gym.make(args.env_name)
    
    num_actions = env.action_space.n
    num_states = env.observation_space.shape[0]
    
    # Create Neural Networks
    lr_policy = args.lr
    lr_value = args.lr_c
    
    policy = NeuralNetwork([num_states, *args.hidden_layers, num_actions],
                           [*len(args.hidden_layers)*[Activation.relu], Activation.softmax])
    
    policy_optimiser = Optimiser.Momentum(policy, batch_size=args.batch_size)

    args.models = [policy]
    
    if args.baseline:
        value = NeuralNetwork([num_states, *args.hidden_layers, 1],
                              [*len(args.hidden_layers)*[Activation.relu], Activation.linear])
        
        value_optimiser = Optimiser.Momentum(value, loss_fcn=Loss.mean_squared_error, batch_size=args.batch_size)

        args.models.append(value)
    
    states = []
    actions = []
    action_probs = []
    G = []
#    gamma_discount = []
    
    cart_factor = 1
    
    episode_num = 0
    while episode_num < args.num_episodes: # for loop does not handle changing of num_episodes
        episode_num += 1
        
        done = False
        state = env.reset()
        state = np.expand_dims(state, axis=1)
        
        rewards = []
        clipped_rewards = []
        
        while not done:
            if args.render: env.render()
            states.append(state)
    
            action_prob = np.squeeze(policy.predict(state))
            action = np.random.choice(num_actions, p=action_prob)
    
            actions.append(action)
            
            state, reward, done, _ = env.step(action)
            state = np.expand_dims(state, axis=1)
    
            rewards.append(reward)
            action_probs.append(action_prob)

#            gamma_discount.append(gamma**i)
            
            clipped_rewards.append(reward)
            
        ep_reward = np.sum(rewards)
        
        # Moving average of the reward
        args.mean_reward_deque.append(ep_reward)
        mean_reward = np.mean(args.mean_reward_deque)
        
        print("({}) mean reward {:5.2f} reward: {:5.2f}".format(episode_num, mean_reward, ep_reward))
        
        if args.log_dir is not None:
            args.reward_logger.append([ep_reward])
            if episode_num % 50 == 0:
                args.reward_logger.save()
                
        G.extend(discount_rewards(clipped_rewards, args.gamma))
        
        # Update Policy and Baseline Networks
        if episode_num % args.episode_batch_size == 0:
            
            states = np.hstack(states)
            
            init_gradient = Loss.cross_entropy_loss_deriv(np.array(actions), np.vstack(action_probs).T)
            
            if args.env_name == 'CartPole-v1': cart_factor = 1/len(actions)
            if args.baseline:
                values = value.predict(states)
            
                td_errors = G - values
                init_gradient = init_gradient*td_errors
                value_optimiser.train(states, np.expand_dims(G, axis=0),learning_rate=lr_value*cart_factor)
            else:
                init_gradient = init_gradient*np.expand_dims(G,axis=0)
            policy_optimiser.train(states, np.array(actions), learning_rate=lr_policy*cart_factor, init_gradient=init_gradient)

            G = []
            states = []
            actions = []
            action_probs = []
#            gamma_discount = []
    
    env.close()
