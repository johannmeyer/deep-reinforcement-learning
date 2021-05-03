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

def GAE(args):
    
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
    policy_optimiser.act_loss_merged = True

    value = NeuralNetwork([num_states, *args.hidden_layers, 1],
                              [*len(args.hidden_layers)*[Activation.relu], Activation.linear])
        
    value_optimiser = Optimiser.Momentum(value, loss_fcn=Loss.mean_squared_error, batch_size=args.batch_size)

    args.models = [policy, value]
    
    mb_states = np.array([]).reshape(num_states,0)
    mb_next_states = np.array([]).reshape(num_states,0)
    mb_actions = []
    mb_action_probs = []
#    rewards = []
    mb_gae = np.array([[]])
    dones = []
    mb_returns = np.array([[]])

#    gamma_discount = []
    
    cart_factor = 1
    
    episode_num = 0
    while episode_num < args.num_episodes: # for loop does not handle changing of num_episodes
        episode_num += 1
        
        done = False
        state = env.reset()
        state = np.expand_dims(state, axis=1)
        
        ep_reward = 0
        
        states = []
        next_states = []
        clipped_rewards = []
        dones = []
        
        while not done:
            if args.render: env.render()
            states.append(state)
    
            action_prob = np.squeeze(policy.predict(state))
            action = np.random.choice(num_actions, p=action_prob)
    
            mb_actions.append(action)
            
            state, reward, done, _ = env.step(action)
            state = np.expand_dims(state, axis=1)
    
#            rewards.append(reward)
            mb_action_probs.append(action_prob)

#            gamma_discount.append(gamma**i)
            next_states.append(state)
            dones.append(done)
            
#            if reward < -1:
#                clipped_rewards.append(-1)
#            elif reward >= 100:
#                clipped_rewards.append(1)
#            elif reward > 0.5:
#                clipped_rewards.append(0.5)
#            else:
            clipped_rewards.append(reward)
            
            ep_reward += reward
        
        # Moving average of the reward
        args.mean_reward_deque.append(ep_reward)
        mean_reward = np.mean(args.mean_reward_deque)
        
        print("({}) mean reward {:5.2f} reward: {:5.2f}".format(episode_num, mean_reward, ep_reward))
        
        if args.log_dir is not None:
            args.reward_logger.append([ep_reward])
            if episode_num % 50 == 0:
                args.reward_logger.save()
                
        states = np.hstack(states)
        next_states = np.hstack(next_states)        
        values = value.predict(states)
        next_values = value.predict(next_states)   
        
        target = np.array(clipped_rewards) + args.gamma*next_values*(1-np.array(dones))
        td_errors = target - values
        
        mb_gae = np.hstack((mb_gae, discount_rewards(td_errors.T, args.gamma*args.Lambda).T))
        mb_returns = np.hstack((mb_returns, np.expand_dims(discount_rewards(np.array(clipped_rewards), args.gamma), axis=0)))
    #    rewards = np.clip(rewards, -1, 1)
        mb_states = np.hstack((mb_states, states))
        mb_next_states = np.hstack((mb_next_states , next_states))
        
        # Update Policy and Baseline Networks
        if episode_num % args.episode_batch_size == 0:

            init_gradient = Loss.cross_entropy_loss_deriv(np.array(mb_actions), np.vstack(mb_action_probs).T)

            init_gradient = init_gradient*mb_gae
            
            if args.env_name == 'CartPole-v1': cart_factor = 1/len(mb_actions)
            
            value_optimiser.train(mb_states, mb_returns, learning_rate=lr_value*cart_factor)
            
            policy_optimiser.train(mb_states, None, learning_rate=lr_policy*cart_factor, init_gradient=init_gradient)

            mb_states = np.array([]).reshape(num_states,0)
            mb_next_states = np.array([]).reshape(num_states,0)
            mb_gae = np.array([[]])
            mb_actions = []
            mb_action_probs = []
            dones = []
            mb_returns = np.array([[]])
#            gamma_discount = []
    
    env.close()
