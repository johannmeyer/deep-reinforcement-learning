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

def A2C(args):
    
    # Initialise Environment
    env = gym.make(args.env_name)
#    env.seed(0) # TODO
#    np.random.seed(0) #TODO
    
    num_actions = env.action_space.n
    num_states = env.observation_space.shape[0]
    
    # Create Neural Networks
    lr_policy = args.lr
    lr_value = args.lr_c
    
    policy = NeuralNetwork([num_states, *args.hidden_layers, num_actions],
                           [*len(args.hidden_layers)*[Activation.relu], Activation.softmax])
    
    policy_optimiser = Optimiser.Momentum(policy, batch_size=args.batch_size)

    value = NeuralNetwork([num_states, *args.hidden_layers, 1],
                              [*len(args.hidden_layers)*[Activation.relu], Activation.linear])
        
    value_optimiser = Optimiser.Momentum(value, loss_fcn=Loss.mean_squared_error, batch_size=args.batch_size)

    args.models = [policy, value]
    
    mb_states = []
    mb_next_states = []
    mb_actions = []
    mb_action_probs = []
    mb_clipped_rewards = []
    mb_dones = []
#    gamma_discount = []
    
    episode_num = 0
    while episode_num < args.num_episodes: # for loop does not handle changing of num_episodes
        episode_num += 1
        
        done = False
        state = env.reset()
        state = np.expand_dims(state, axis=1)
        
        ep_reward = 0
        
        while not done:
            if args.render: env.render()
            mb_states.append(state)
    
            action_prob = np.squeeze(policy.predict(state))
            action = np.random.choice(num_actions, p=action_prob)
    
            mb_actions.append(action)
            
            state, reward, done, _ = env.step(action)
            state = np.expand_dims(state, axis=1)
    
#            rewards.append(reward)
            mb_action_probs.append(action_prob)
            # TODO
#            gamma_discount.append(gamma**i)
            mb_next_states.append(state)
            mb_dones.append(done)
            
#            if reward < -1:
#                clipped_rewards.append(-1)
#            elif reward >= 100:
#                clipped_rewards.append(1)
#            elif reward > 1:
#                clipped_rewards.append(0.5)
#            else:
            mb_clipped_rewards.append(reward)
            
            ep_reward += reward
        
        # Moving average of the reward
        args.mean_reward_deque.append(ep_reward)
        mean_reward = np.mean(args.mean_reward_deque)
        
        print("({}) mean reward {:5.2f} reward: {:5.2f}".format(episode_num, mean_reward, ep_reward))
        
        if args.log_dir is not None:
            args.reward_logger.append([ep_reward])
            if episode_num % 50 == 0:
                args.reward_logger.save()
  

    #    rewards = np.clip(rewards, -1, 1)
        
        
        # Update Policy and Baseline Networks
        if episode_num % args.episode_batch_size == 0:
            mb_states = np.hstack(mb_states)
            mb_next_states = np.hstack(mb_next_states)
            
            values = value.predict(mb_states)
            next_values = value.predict(mb_next_states)   
            
            init_gradient = Loss.cross_entropy_loss_deriv(np.array(mb_actions), np.vstack(mb_action_probs).T)

            mb_targets = np.array(mb_clipped_rewards) + args.gamma*next_values*(1-np.array(mb_dones))
            td_errors = mb_targets - values

            init_gradient = init_gradient*td_errors
            value_optimiser.train(mb_states, mb_targets,learning_rate=lr_value/len(mb_actions))
            
            policy_optimiser.train(mb_states, np.array(mb_actions), learning_rate=lr_policy/len(mb_actions), init_gradient=init_gradient)

            mb_states = []
            mb_next_states = []
            mb_actions = []
            mb_action_probs = []
            mb_dones = []
            mb_clipped_rewards = []
    
    env.close()
